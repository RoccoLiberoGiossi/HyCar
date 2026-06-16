import numpy as np

class ElectricMotor:
    """
    IPMSM traction motor with simplified 2-D efficiency map.
    Efficiency is a function of power fraction relative to peak.
    Peak efficiency at 60–80% of rated power; drops at low load
    (iron loss dominance) and high load (copper loss dominance).
    Includes continuous and peak (30 s) power ratings.
    """
    def __init__(self, continuous_power_kw: float, peak_power_kw: float,
                 peak_duration_s: float = 25.0,
                 peak_efficiency: float = 0.965):
        self.continuous_power_kw = continuous_power_kw
        self.peak_power_kw = peak_power_kw
        self.peak_duration_s = peak_duration_s
        self.peak_efficiency = peak_efficiency

        # Peak power usage timer
        self._peak_timer_s = 0.0
        self._in_peak_mode = False

    def _efficiency(self, power_fraction: float) -> float:
        """
        Approximate IPMSM efficiency vs. load curve.
        Peak efficiency near 70% load. Drops at extremes.
        """
        if power_fraction <= 0.0:
            return 0.0  # Regen handled separately
        # Quadratic approximation centred at 0.70 of rated load
        penalty = 1.0 - 0.08 * (power_fraction - 0.70) ** 2
        penalty = max(0.82, min(1.0, penalty))
        return self.peak_efficiency * penalty

    def request(self, mech_power_kw: float, delta_time_s: float) -> dict:
        """
        Args:
            mech_power_kw: Mechanical power needed at wheel side.
                           Positive = motoring, negative = regenerating.
        Returns: electrical power consumed/returned [kW], motor efficiency
        """
        is_regen = mech_power_kw < 0

        if is_regen:
            # Regen: motor acts as generator
            # Electrical power returned = mech power × efficiency
            regen_mech = abs(mech_power_kw)
            power_fraction = regen_mech / self.continuous_power_kw
            eff = self._efficiency(power_fraction)
            electrical_kw = -(regen_mech * eff)  # Negative = charging
            self._peak_timer_s = max(0.0, self._peak_timer_s - delta_time_s)
            return {"electrical_kw": electrical_kw, "efficiency": eff, "peak_mode": False}

        # Motoring: enforce peak power time limit
        limit_kw = self.continuous_power_kw
        if mech_power_kw > self.continuous_power_kw:
            if self._peak_timer_s < self.peak_duration_s:
                limit_kw = self.peak_power_kw
                self._peak_timer_s += delta_time_s
                self._in_peak_mode = True
            else:
                limit_kw = self.continuous_power_kw  # Thermal derating
                self._in_peak_mode = False
        else:
            self._peak_timer_s = max(0.0, self._peak_timer_s - delta_time_s * 0.5)
            self._in_peak_mode = False

        actual_mech_kw = min(mech_power_kw, limit_kw)
        power_fraction = actual_mech_kw / self.continuous_power_kw
        eff = self._efficiency(power_fraction)

        electrical_kw = actual_mech_kw / eff if eff > 0 else actual_mech_kw

        return {
            "electrical_kw": electrical_kw,
            "efficiency": eff,
            "peak_mode": self._in_peak_mode,
            "limited": actual_mech_kw < mech_power_kw,
        }
    
class Inverter:
    """
    3-phase SiC voltage-source inverter model.
    Efficiency is load-dependent: high at mid-load, lower at very light load
    (switching losses dominate) and slightly reduced at peak (conduction losses).
    Regen limit tied to battery max charge acceptance rather than hardcoded.
    """
    PEAK_EFFICIENCY = 0.990          # SiC at optimal load point (~60% rated)
    NO_LOAD_EFFICIENCY = 0.930       # Very light load: switching loss penalty
    SWITCHING_FREQ_KHZ = 20          # Informational; affects loss shape

    def __init__(self, rated_power_kw: float, max_regen_kw: float = None):
        self.rated_power_kw = rated_power_kw
        self.max_regen_kw = max_regen_kw or rated_power_kw * 0.65

    def _efficiency(self, power_fraction: float) -> float:
        """
        Load-dependent efficiency curve for SiC inverter.
        Minimum at no-load, peak at ~0.5–0.7 of rated, slight drop at 1.0.
        """
        if power_fraction <= 0.0:
            return self.NO_LOAD_EFFICIENCY
        # Blend: rises quickly from no-load, slight droop at full load
        eff = self.NO_LOAD_EFFICIENCY + \
              (self.PEAK_EFFICIENCY - self.NO_LOAD_EFFICIENCY) * \
              (1.0 - np.exp(-8.0 * power_fraction))
        # Small high-load droop due to conduction losses
        if power_fraction > 0.70:
            eff -= 0.005 * (power_fraction - 0.70)
        return max(self.NO_LOAD_EFFICIENCY, min(self.PEAK_EFFICIENCY, eff))

    def process(self, motor_electrical_kw: float) -> dict:
        """
        Args:
            motor_electrical_kw: Power at AC motor terminals.
                Positive = motoring (inverter draws from DC bus).
                Negative = regen (inverter pushes to DC bus).
        Returns: dc_bus_power_kw (positive = draw, negative = return)
        """
        is_regen = motor_electrical_kw < 0
        abs_power = abs(motor_electrical_kw)
        power_fraction = abs_power / self.rated_power_kw

        eff = self._efficiency(power_fraction)

        if is_regen:
            clamped = min(abs_power, self.max_regen_kw)
            dc_power = -(clamped * eff)      # Power returned to DC bus
        else:
            dc_power = abs_power / eff        # DC bus must supply more than AC output

        return {
            "dc_bus_kw": dc_power,
            "efficiency": eff,
            "regen_limited": is_regen and abs_power > self.max_regen_kw,
        }