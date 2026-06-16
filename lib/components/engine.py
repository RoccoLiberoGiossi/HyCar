class InternalCombustionEngine:
    """
    Series hybrid range extender — generator duty only.
    Operates along a single-speed generator line (fixed RPM, variable throttle).
    Uses a BSFC map approximated by a quadratic bowl centred on the optimum point.
    Includes warm-up penalty and startup/shutdown transient modelling.
    """
    FUEL_LHV_J_PER_KG = 43_400_000          # Gasoline lower heating value
    FUEL_DENSITY_KG_PER_L = 0.745
    WARMUP_DURATION_S = 90                   # Time to reach full thermal efficiency
    WARMUP_EFFICIENCY_FACTOR = 0.65          # Efficiency multiplier at cold start

    def __init__(self, max_power_kw: float,
                 optimum_power_kw: float = None,
                 peak_efficiency: float = 0.36):
        self.max_power_kw = max_power_kw
        # Optimum power: ~65–75% of max for best BSFC
        self.optimum_power_kw = optimum_power_kw or (max_power_kw * 0.70)
        self.peak_efficiency = peak_efficiency  # 36% is realistic for REx class

        # State
        self.is_running = False
        self.warmup_timer_s = 0.0
        self.startup_delay_s = 1.5           # Crank-to-fire time

    def _bsfc_efficiency(self, power_kw: float) -> float:
        """
        Approximate BSFC island as a quadratic penalty around the optimum point.
        Efficiency drops off-peak due to pumping losses (under-load) and
        thermal/friction losses (over-load).
        """
        if power_kw <= 0:
            return 0.0
        ratio = power_kw / self.optimum_power_kw
        # Quadratic bowl: 1.0 at optimum, degrades symmetrically
        penalty = 1.0 - 0.35 * (ratio - 1.0) ** 2
        penalty = max(0.55, min(1.0, penalty))    # Clamp: never below 55% of peak
        return self.peak_efficiency * penalty

    def _warmup_factor(self) -> float:
        """Linear ramp from cold-start penalty to full efficiency."""
        if self.warmup_timer_s >= self.WARMUP_DURATION_S:
            return 1.0
        progress = self.warmup_timer_s / self.WARMUP_DURATION_S
        return self.WARMUP_EFFICIENCY_FACTOR + (1.0 - self.WARMUP_EFFICIENCY_FACTOR) * progress

    def start(self):
        self.is_running = True
        # Keep warmup timer — partial warm restarts are more efficient
        # (don't reset to 0 on every restart)

    def stop(self):
        self.is_running = False

    def step(self, requested_charge_power_kw: float, delta_time_s: float) -> dict:
        """
        Called every timestep. Returns actual electrical output and fuel consumed.

        Args:
            requested_charge_power_kw: Power the EMS wants from the generator [kW]
            delta_time_s: Timestep duration [s]
        """
        if not self.is_running:
            self.warmup_timer_s -= delta_time_s if self.warmup_timer_s > 0 else 0
            return {"power_kw": 0.0, "fuel_flow_gps": 0.0, "fuel_burned_g": 0.0}

        self.warmup_timer_s += delta_time_s

        # Clamp requested power to physical limits
        actual_power_kw = max(0.0, min(self.max_power_kw, requested_charge_power_kw))

        # Efficiency from BSFC map, modified by warm-up state
        thermal_efficiency = self._bsfc_efficiency(actual_power_kw) * self._warmup_factor()

        # Fuel power required = shaft power / efficiency
        if thermal_efficiency > 0:
            fuel_power_w = (actual_power_kw * 1000.0) / thermal_efficiency
        else:
            fuel_power_w = 0.0

        fuel_flow_kg_per_s = fuel_power_w / self.FUEL_LHV_J_PER_KG
        fuel_burned_g = fuel_flow_kg_per_s * delta_time_s * 1000.0

        return {
            "power_kw": actual_power_kw,
            "fuel_flow_gps": fuel_flow_kg_per_s * 1000.0,
            "fuel_burned_g": fuel_burned_g,
            "efficiency": thermal_efficiency,
        }