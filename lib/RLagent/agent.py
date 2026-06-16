import gymnasium as gym
from gymnasium import spaces

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from lib.settings import *

from lib.components.batteries import ElectricBattery
from lib.components.duty_cicle import StochasticDriveCycle
from lib.components.motor_inverter import ElectricMotor, Inverter
from lib.components.engine import InternalCombustionEngine
from lib.components.batteries import ElectricBattery
from lib.components.vehicles import Vehicle


class HybridVehicleEnv(gym.Env):
    """Custom Environment for a hybrid vehicle energy management problem."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(HybridVehicleEnv, self).__init__()

        # --------------------------------------------------------------- #
        # Agent parameters
        # --------------------------------------------------------------- #

        # Action space: 0 = electric, 1 = ICE recharge
        self.action_space = spaces.Discrete(2)

        # Observation space: [distance_traveled_m, soc, gas_liters,
        #                     speed_kmh, accel_mps2, ice warmup timer]
        # Normalizations: [0, 2e6], [0, 1], [0, MAX_GAS_L], [0, 250], [-5, 5], [0, WARMUP_DURATION_S]
        low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --------------------------------------------------------------- #
        # System definition
        # --------------------------------------------------------------- #

        # is done flag for episode termination
        self.done = False

        self.last_printed_milestone = 0

        # critical conditions for episode termination
        self.max_gas_g = (
            MAX_GAS_L * InternalCombustionEngine.FUEL_DENSITY_KG_PER_L * 1000
        )
        self.min_soc = MIN_ELECTRIC_SOC

        # Instantiate components
        self.battery, self.ice, self.motor, self.inverter, self.vehicle, self.cycle = (
            self.initialize_components()
        )

        # Observation variables initialization
        self.total_distance_m = 0.0
        self.soc = self.battery.soc
        self.gas_liters = 0.0
        self.speed_kmh = self.cycle.speed_kmh
        self.accel_mps2 = self.cycle.accel_mps2
        self.ice_warmup = self.ice.warmup_timer_s

        self.total_fuel_g = 0.0
        self.total_regen_kwh = 0.0
        self.total_motor_kwh = 0.0
        self.ice_on_seconds = 0
        self.t = 0

        # Telemetry lists (downsampled)
        self.telemetry = {
            "time_h": [],
            "dist_km": [],
            "speed_kmh": [],
            "soc_pct": [],
            "p_motor_kw": [],  # Electrical power drawn from / returned to DC bus by motor path
            "p_ice_kw": [],  # ICE generation power
            "p_regen_kw": [],  # Regen power (negative convention, for separate fill plot)
            "fuel_l": [],  # Cumulative fuel in litres
            "motor_eff": [],
            "inv_eff": [],
            "ice_eff": [],
            "bat_heat_w": [],
            "env": [],
            "reward": [],
            "vehicle mass_kg": [],
        }

        self.reset()

    def initialize_components(self):

        battery = ElectricBattery(
            capacity_kwh=BATTERY_CAPACITY_KWH,
            initial_soc=INITIAL_SOC,
            internal_resistance_ohm=INTERNAL_RESISTANCE_OHM,
        )

        ice = InternalCombustionEngine(
            max_power_kw=ICE_MAX_POWER_KW,
            optimum_power_kw=ICE_OPTIMUM_POWER_KW,
            peak_efficiency=ICE_PEAK_EFFICIENCY,
        )

        motor = ElectricMotor(
            continuous_power_kw=MOTOR_CONT_POWER_KW,
            peak_power_kw=MOTOR_PEAK_POWER_KW,
            peak_duration_s=PEAK_DURATION_S,
            peak_efficiency=MOTOR_PEAK_EFFICIENCY,
        )

        inverter = Inverter(
            rated_power_kw=INVERTER_RATED_KW,
            max_regen_kw=INVERTER_RATED_KW * INVERTER_REGEN_FACTOR,
        )

        vehicle = Vehicle(
            mass_kg=VEHICLE_MASS_KG + self.max_gas_g / 1000.0,
            cd=DRAG_COEFFICIENT,
            frontal_area_m2=FRONTAL_AREA_M2,
            crr=ROLLING_RESISTANCE,
        )

        cycle = StochasticDriveCycle(
            initial_speed_kmh=INITIAL_SPEED_KMH, initial_env=ROAD_TYPE
        )

        return battery, ice, motor, inverter, vehicle, cycle

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.battery, self.ice, self.motor, self.inverter, self.vehicle, self.cycle = (
            self.initialize_components()
        )

        # Observation variables initialization
        self.total_distance_m = 0.0
        self.total_distance_km = 0.0
        self.soc = self.battery.soc
        self.gas_liters = 0.0
        self.speed_kmh = self.cycle.speed_kmh
        self.accel_mps2 = self.cycle.accel_mps2
        self.ice_warmup = self.ice.warmup_timer_s

        self.total_fuel_g = 0.0
        self.total_regen_kwh = 0.0
        self.total_motor_kwh = 0.0
        self.ice_on_seconds = 0
        self.t = 0

        # is done flag for episode termination
        self.done = False

        self.last_printed_milestone = 0

        # Clear telemetry
        for key in self.telemetry:
            self.telemetry[key] = []

        return self._get_observation(), {}
    
    def _get_observation(self):
        return [
            self.total_distance_km / TARGET_DISTANCE_M * 1000.0,
            self.soc,
            self.gas_liters / MAX_GAS_L,
            self.speed_kmh / 250.0,
            (self.accel_mps2) / 10.0,  # Normalize to [-1,1] with -5 to +5 m/s² range
            self.ice_warmup / self.ice.WARMUP_DURATION_S,
        ]
    
    def _calculate_reward(self):
        # 1. Base log curve design
        # Adding 1 guarantees that when distance is 0, log(1) = 0.0
        # Scaling factor 0.1 controls how fast the curve transitions from steep to flat
        gamma = 0.1
        reward = math.log(gamma * self.total_distance_km + 1)

        return reward

    def _check_done(self):
        # if self.gas_liters >= MAX_GAS_L and self.soc <= self.min_soc:
        #     return True
        if self.soc <= self.min_soc:
            return True
        # Fuel exhausted with no electric reserve left
        if self.gas_liters >= MAX_GAS_L and self.soc < 0.15:
            return True
        return False
    
    def step(self, action):

        self.t += DT_S
        
        # 1. Drive cycle step - get new speed and acceleration
        self.speed_kmh, self.accel_mps2 = self.cycle.get_next_step(DT_S)
        self.total_distance_m += (self.speed_kmh / 3.6) * DT_S

        # 2. Road load → mechanical power needed at the wheel
        mech_power_kw = self.vehicle.road_load_kw(self.speed_kmh, self.accel_mps2)

        # 3. Motor converts mechanical demand to electrical demand
        motor_result = self.motor.request(mech_power_kw, DT_S)
        motor_elec_kw = motor_result["electrical_kw"]
        # Accumulate energy stats
        if motor_elec_kw >= 0:
            self.total_motor_kwh += motor_elec_kw * (DT_S / 3600.0)
        else:
            self.total_regen_kwh += abs(motor_elec_kw) * (DT_S / 3600.0)

        # 4. Inverter processes the AC↔DC conversion
        inv_result = self.inverter.process(motor_elec_kw)
        dc_bus_demand_kw = inv_result["dc_bus_kw"]

        # RL action on ICE
        if action == 1 and self.gas_liters < MAX_GAS_L:
            self.ice.start()
        elif action == 0:
            self.ice.stop()
        else:
            self.ice.stop()

        # 6. ICE generation step
        if self.ice.is_running:
            ice_result = self.ice.step(ICE_OPTIMUM_POWER_KW, DT_S)
            self.ice_on_seconds += DT_S
        else:
            _ = self.ice.step(0.0, DT_S)
            ice_result = {"power_kw": 0.0, "fuel_burned_g": 0.0, "efficiency": 0.0}

        self.total_fuel_g += ice_result["fuel_burned_g"]
        self.vehicle.update_mass(VEHICLE_MASS_KG, self.max_gas_g / 1000.0, self.total_fuel_g / 1000.0)

        # 7. Net power on DC bus → battery
        #    Positive net = battery discharging; negative net = battery charging
        net_battery_kw = dc_bus_demand_kw - ice_result["power_kw"]
        bat_result = self.battery.update(net_battery_kw, DT_S)

        self.total_distance_km = self.total_distance_m / 1000.0
        self.soc = self.battery.soc
        self.gas_liters = self.total_fuel_g / (InternalCombustionEngine.FUEL_DENSITY_KG_PER_L * 1000)
        self.ice_warmup = self.ice.warmup_timer_s

        reward = self._calculate_reward()
        self.done = self._check_done()

        if self.t % LOG_INTERVAL_S == 0:
            self.telemetry["time_h"].append(self.t / 3600.0)
            self.telemetry["dist_km"].append(self.total_distance_km)
            self.telemetry["speed_kmh"].append(self.speed_kmh)
            self.telemetry["soc_pct"].append(self.soc * 100.0)
            self.telemetry["p_motor_kw"].append(motor_elec_kw)
            self.telemetry["p_ice_kw"].append(ice_result["power_kw"])
            self.telemetry["p_regen_kw"].append(-motor_elec_kw if motor_elec_kw < 0 else 0.0)
            self.telemetry["fuel_l"].append(self.gas_liters)
            self.telemetry["motor_eff"].append(motor_result["efficiency"])
            self.telemetry["inv_eff"].append(inv_result["efficiency"])
            self.telemetry["ice_eff"].append(ice_result["efficiency"])
            self.telemetry["bat_heat_w"].append(bat_result["heat_loss_w"])
            self.telemetry["env"].append(self.cycle.current_env)
            self.telemetry["reward"].append(reward)
            self.telemetry["vehicle mass_kg"].append(self.vehicle.mass_kg)

        # Calculate what the current milestone bracket is
        current_milestone = int(self.total_distance_km // 100) * 100

        # Only print if we have moved into a NEW milestone bracket
        if current_milestone > self.last_printed_milestone:
            print(f"Distance: {self.total_distance_km:.1f} km, SoC: {self.soc*100:.1f}%, Gas: {self.gas_liters:.2f} L, Reward: {reward:.3f}, action: {action}")
            self.last_printed_milestone = current_milestone  # Latch the state

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # ------------------------------------------------------------------
        # Arrays
        # ------------------------------------------------------------------
        dist      = np.array(self.telemetry["dist_km"])
        speed     = np.array(self.telemetry["speed_kmh"])
        soc       = np.array(self.telemetry["soc_pct"])
        p_motor   = np.array(self.telemetry["p_motor_kw"])
        p_ice     = np.array(self.telemetry["p_ice_kw"])
        fuel      = np.array(self.telemetry["fuel_l"])
        ice_eff   = np.array(self.telemetry["ice_eff"])
        motor_eff = np.array(self.telemetry["motor_eff"])
        inv_eff   = np.array(self.telemetry["inv_eff"])
        bat_heat  = np.array(self.telemetry["bat_heat_w"])
        env_arr   = np.array(self.telemetry["env"])

        env_colors = {'CITY': '#e8eeff', 'SUBURBAN': '#e8f8ee', 'HIGHWAY': '#fff4e8'}

        # ------------------------------------------------------------------
        # Figure — 4 rows × 2 columns
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(16, 20))
        fig.patch.set_facecolor('#f7f7f7')

        gs = gridspec.GridSpec(
            4, 2,
            figure=fig,
            hspace=0.40,
            wspace=0.28,
            left=0.07, right=0.97,
            top=0.94, bottom=0.05,
        )

        # ------------------------------------------------------------------
        # Helper: shade driving environment bands on any axis
        # ------------------------------------------------------------------
        def shade_env(ax):
            if len(env_arr) < 2:
                return
            prev = env_arr[0]
            x0   = dist[0]
            for i in range(1, len(dist)):
                if env_arr[i] != prev or i == len(dist) - 1:
                    ax.axvspan(x0, dist[i], color=env_colors[prev], alpha=0.6, linewidth=0)
                    prev = env_arr[i]
                    x0   = dist[i]

        env_legend = [
            Patch(facecolor=env_colors['CITY'],     label='City',     alpha=0.85),
            Patch(facecolor=env_colors['SUBURBAN'], label='Suburban', alpha=0.85),
            Patch(facecolor=env_colors['HIGHWAY'],  label='Highway',  alpha=0.85),
        ]

        def style(ax, title, ylabel, xlabel=None, xlim=None):
            ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=6)
            ax.set_ylabel(ylabel, fontsize=9)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=9)
            if xlim:
                ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.22, linewidth=0.6)
            ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.set_facecolor('white')

        xlim = (dist[0], dist[-1])

        # ── R0 C0 : Speed ────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        shade_env(ax)
        ax.plot(dist, speed, color='#3b1f8c', linewidth=0.55, alpha=0.9)
        style(ax, "Speed profile", "Speed (km/h)", xlim=xlim)
        ax.legend(handles=env_legend, fontsize=7.5, ncol=3,
                loc='upper right', framealpha=0.85)

        # ── R0 C1 : Battery SoC ──────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(dist, soc, color='#0a7c5e', linewidth=1.6, zorder=3, label='SoC')
        ax.axhline(REX_TRIGGER_SOC * 100, color='#c0392b', ls='--', lw=1.1, alpha=0.9,
                label=f'REx on  {REX_TRIGGER_SOC*100:.0f}%')
        ax.axhline(REX_CUTOFF_SOC  * 100, color='#1a5276', ls='--', lw=1.1, alpha=0.9,
                label=f'REx off {REX_CUTOFF_SOC*100:.0f}%')
        ax.fill_between(dist, soc, REX_TRIGGER_SOC * 100,
                        where=(soc < REX_TRIGGER_SOC * 100),
                        color='#e74c3c', alpha=0.18, zorder=2, label='ICE active')
        style(ax, "Battery SoC — EMS hysteresis", "SoC (%)", xlim=xlim)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7.5, ncol=2, loc='lower left', framealpha=0.85)

        # ── R1 C0 : Power flow ───────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(dist, p_motor, color='#5b4fcf', lw=0.5, alpha=0.50, label='Motor DC demand')
        ax.plot(dist, p_ice,   color='#c0392b', lw=1.6, alpha=0.95, label='ICE generation')
        mask_regen = p_motor < 0
        ax.fill_between(dist, p_motor, 0, where=mask_regen,
                        color='#27ae60', alpha=0.40, label='Regen')
        ax.axhline(0, color='#666', lw=0.6)
        style(ax, "Power flow", "Power (kW)", xlim=xlim)
        ax.legend(fontsize=7.5, ncol=3, loc='upper left', framealpha=0.85)

        # ── R1 C1 : Cumulative fuel ──────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(dist, fuel, color='#8b3a00', lw=2.0, label='Fuel consumed')
        ax.fill_between(dist, fuel, color='#8b3a00', alpha=0.10)
        style(ax, "Cumulative fuel", "Volume (L)", xlim=xlim)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)

        # ── R2 C0 : Motor efficiency ─────────────────────────────────────
        ax = fig.add_subplot(gs[2, 0])
        mask_mot = motor_eff > 0
        ax.scatter(dist[mask_mot], motor_eff[mask_mot]*100,
                s=0.6, c='#2563eb', alpha=0.25, rasterized=True)
        # Rolling mean for readability
        if mask_mot.sum() > 50:
            w = 80
            rm = np.convolve(motor_eff[mask_mot]*100, np.ones(w)/w, mode='valid')
            rd = dist[mask_mot][w//2 : w//2 + len(rm)]
            ax.plot(rd, rm, color='#1e3a8a', lw=1.4, label='80-pt rolling mean')
            ax.legend(fontsize=7.5, loc='lower right', framealpha=0.85)
        style(ax, "Motor efficiency", "η (%)", xlim=xlim)
        ax.set_ylim(78, 100)

        # ── R2 C1 : Inverter efficiency ──────────────────────────────────
        ax = fig.add_subplot(gs[2, 1])
        mask_inv = inv_eff > 0
        ax.scatter(dist[mask_inv], inv_eff[mask_inv]*100,
                s=0.6, c='#7c3aed', alpha=0.25, rasterized=True)
        if mask_inv.sum() > 50:
            w = 80
            rm = np.convolve(inv_eff[mask_inv]*100, np.ones(w)/w, mode='valid')
            rd = dist[mask_inv][w//2 : w//2 + len(rm)]
            ax.plot(rd, rm, color='#4c1d95', lw=1.4, label='80-pt rolling mean')
            ax.legend(fontsize=7.5, loc='lower right', framealpha=0.85)
        style(ax, "Inverter (SiC) efficiency", "η (%)", xlim=xlim)
        ax.set_ylim(88, 100)

        # ── R3 C0 : ICE thermal efficiency ──────────────────────────────
        ax = fig.add_subplot(gs[3, 0])
        mask_ice = ice_eff > 0
        if mask_ice.sum() > 0:
            ax.scatter(dist[mask_ice], ice_eff[mask_ice]*100,
                    s=1.0, c='#b91c1c', alpha=0.40, rasterized=True,
                    label='ICE η (active)')
            ax.legend(fontsize=7.5, loc='lower right', framealpha=0.85)
        else:
            ax.text(0.5, 0.5, 'ICE never active', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#888')
        style(ax, "ICE thermal efficiency (active only)", "η (%)",
            xlabel="Distance (km)", xlim=xlim)
        ax.set_ylim(0, 42)

        # ── R3 C1 : Battery heat dissipation ────────────────────────────
        ax = fig.add_subplot(gs[3, 1])
        ax.plot(dist, bat_heat, color='#d97706', lw=0.7, alpha=0.75, label='Instantaneous')
        ax.fill_between(dist, bat_heat, color='#d97706', alpha=0.15)
        # Cumulative on secondary y-axis
        ax2 = ax.twinx()
        bat_heat_cumwh = np.cumsum(bat_heat * (5.0 / 3600.0))   # 5 s log interval
        ax2.plot(dist, bat_heat_cumwh, color='#92400e', lw=1.4, ls='--', label='Cumulative (Wh)')
        ax2.set_ylabel("Cumulative heat (Wh)", fontsize=8, color='#92400e')
        ax2.tick_params(labelsize=8, colors='#92400e')
        style(ax, "Battery Rint heat dissipation", "Instantaneous loss (W)",
            xlabel="Distance (km)", xlim=xlim)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                fontsize=7.5, loc='upper left', framealpha=0.85)

        plt.show()
