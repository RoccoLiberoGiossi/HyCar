import numpy as np

class Vehicle:
    """Road-load model. Separated from motor so physics are clean."""
    def __init__(self, mass_kg: float, cd: float = 0.24,
                 frontal_area_m2: float = 2.3, crr: float = 0.012):
        self.mass_kg = mass_kg
        self.cd = cd
        self.frontal_area_m2 = frontal_area_m2
        self.crr = crr
        self.rho_air = 1.225

    def update_mass(self, vehicle_mass: float, total_fuel_on_start: float, fuel_used: float):
        """
        update vehicle mass based on fuel used. This is called after the ICE step.
        All masses must be provided in kg.
        Args:
            vehicle_mass (float): The base mass of the vehicle without fuel (kg).
            total_fuel_on_start (float): The total mass of fuel at the start of the episode (kg).
            fuel_used (float): The mass of fuel used so far in the episode (kg).
        """
        self.mass_kg = vehicle_mass + total_fuel_on_start - fuel_used

    def road_load_kw(self, speed_kmh: float, accel_mps2: float,
                     grade_deg: float = 0.0) -> float:
        v = speed_kmh / 3.6
        g = 9.81
        f_rolling = self.crr * self.mass_kg * g if v > 0.01 else 0.0
        f_drag = 0.5 * self.rho_air * self.cd * self.frontal_area_m2 * v ** 2
        f_accel = self.mass_kg * accel_mps2
        f_grade = self.mass_kg * g * np.sin(np.radians(grade_deg))
        total_force_n = f_rolling + f_drag + f_accel + f_grade
        return (total_force_n * v) / 1000.0  # kW at the wheels
