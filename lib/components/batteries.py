class ElectricBattery:
    """
    First-order equivalent circuit model (Rint).
    Models coulombic efficiency, internal resistance losses, and
    enforces C-rate and SoC window limits.
    Usable SoC window: 10–90% (protecting cell longevity).
    """
    NOMINAL_VOLTAGE_V = 400.0          # Nominal pack voltage (400 V architecture)
    SOC_MIN = 0.10                     # Hard lower limit (BMS protection)
    SOC_MAX = 0.98                     # Hard upper limit
    SOC_REGEN_LIMIT = 0.95             # Regen cut-off to prevent overcharge
    COULOMBIC_EFFICIENCY_CHARGE = 0.975  # Round-trip coulombic loss on charge path

    def __init__(self, capacity_kwh: float,
                 initial_soc: float = 0.95,
                 internal_resistance_ohm: float = 0.080):
        """
        Args:
            capacity_kwh: Gross (nameplate) pack capacity
            initial_soc: Starting state of charge [0–1]
            internal_resistance_ohm: Pack Rint (typical: 0.05–0.15 Ω for 400V, 85 kWh pack)
        """
        self.capacity_kwh = capacity_kwh
        self.capacity_wh = capacity_kwh * 1000.0
        self.internal_resistance_ohm = internal_resistance_ohm

        self.soc = max(self.SOC_MIN, min(self.SOC_MAX, initial_soc))
        self.energy_stored_wh = self.soc * self.capacity_wh

        # Telemetry
        self.heat_dissipated_wh = 0.0

    @property
    def usable_energy_wh(self):
        usable_range = self.SOC_MAX - self.SOC_MIN
        return (self.soc - self.SOC_MIN) / usable_range * (self.capacity_wh * usable_range)

    def _pack_voltage(self) -> float:
        """
        Linear OCV approximation. Real cells have a non-linear curve
        but linear is adequate for energy simulation (not cell-level).
        Range: ±8% around nominal across SOC window.
        """
        soc_norm = (self.soc - self.SOC_MIN) / (self.SOC_MAX - self.SOC_MIN)
        return self.NOMINAL_VOLTAGE_V * (0.92 + 0.16 * soc_norm)

    def _max_charge_power_kw(self, c_rate_limit: float = 1.5) -> float:
        """Charge power limit decreases above 80% SoC (CC-CV taper)."""
        base_limit = c_rate_limit * self.capacity_kwh
        if self.soc > 0.80:
            # Linear taper from 100% at 80% SoC to 20% at 98% SoC
            taper = 1.0 - ((self.soc - 0.80) / 0.18) * 0.80
            taper = max(0.20, taper)
            return base_limit * taper
        return base_limit

    def _max_discharge_power_kw(self, c_rate_limit: float = 3.0) -> float:
        """Discharge power limit decreases below 15% SoC."""
        base_limit = c_rate_limit * self.capacity_kwh
        if self.soc < 0.15:
            taper = max(0.1, (self.soc - self.SOC_MIN) / 0.05)
            return base_limit * taper
        return base_limit

    def update(self, requested_power_kw: float, delta_time_s: float) -> dict:
        """
        Args:
            requested_power_kw: Positive = discharge (power to motor),
                                 Negative = charge (power from ICE/regen)
        Returns: dict with actual_power_kw, soc, heat_w, limited flag
        """
        ocv = self._pack_voltage()
        is_charging = requested_power_kw < 0

        # Enforce C-rate limits
        if is_charging:
            # Block regen if at or near SoC ceiling
            if self.soc >= self.SOC_REGEN_LIMIT:
                requested_power_kw = 0.0
            max_charge = self._max_charge_power_kw()
            actual_power_kw = max(-max_charge, requested_power_kw)
        else:
            # Block discharge below minimum SoC
            if self.soc <= self.SOC_MIN:
                actual_power_kw = 0.0
            else:
                max_discharge = self._max_discharge_power_kw()
                actual_power_kw = min(max_discharge, requested_power_kw)

        # Internal resistance (Joule) losses: P_loss = I²·R
        # Approximate current: I = P / V_oc
        if ocv > 0:
            current_a = (abs(actual_power_kw) * 1000.0) / ocv
        else:
            current_a = 0.0
        heat_loss_w = current_a ** 2 * self.internal_resistance_ohm
        heat_loss_wh = heat_loss_w * (delta_time_s / 3600.0)
        self.heat_dissipated_wh += heat_loss_wh

        # Net energy change in the cell
        if is_charging:
            # Coulombic efficiency: some charge energy is lost as heat
            energy_delta_wh = -(abs(actual_power_kw) * 1000.0 * delta_time_s / 3600.0) \
                              * self.COULOMBIC_EFFICIENCY_CHARGE
        else:
            energy_delta_wh = actual_power_kw * 1000.0 * delta_time_s / 3600.0

        # Subtract resistive loss from stored energy
        self.energy_stored_wh -= (energy_delta_wh + heat_loss_wh)
        self.energy_stored_wh = max(
            self.SOC_MIN * self.capacity_wh,
            min(self.SOC_MAX * self.capacity_wh, self.energy_stored_wh)
        )
        self.soc = self.energy_stored_wh / self.capacity_wh

        return {
            "actual_power_kw": actual_power_kw,
            "soc": self.soc,
            "heat_loss_w": heat_loss_w,
            "limited": actual_power_kw != requested_power_kw,
        }