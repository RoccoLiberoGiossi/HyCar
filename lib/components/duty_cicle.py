import random
import numpy as np

class StochasticDriveCycle:
    """
    Stochastic drive cycle using an Ornstein-Uhlenbeck process for acceleration,
    producing temporally correlated (smooth) speed profiles that mimic
    real driver behaviour. Mean-reverts toward environment-appropriate
    target speeds.

    OU process: dA = θ(μ - A)dt + σ·dW
      θ = mean reversion speed (driver 'intention' strength)
      μ = target acceleration (derived from speed error)
      σ = volatility (pedal noise / micro-corrections)
    """
    ENV_PARAMS = {
        #            max_spd  target_spd  theta  sigma
        'CITY':     (50.0,    30.0,       0.8,   0.20),
        'SUBURBAN': (90.0,    65.0,       0.5,   0.12),
        'HIGHWAY':  (140.0,   110.0,      0.3,   0.06),
    }

    def __init__(self, initial_speed_kmh: float = 0.0,
                 initial_env: str = 'CITY',
                 time_step_s: float = 1.0):
        self.speed_kmh = initial_speed_kmh
        self.current_env = initial_env
        self.time_step_s = time_step_s
        self.accel_mps2 = 0.0             # Current OU state variable
        self._env_timer_s = random.uniform(300, 900)

        # City stop-light model: periodic forced stops
        self._stop_timer_s = 0.0
        self._in_stop = False
        self._stop_duration_s = 0.0

    def _maybe_switch_env(self, dt: float):
        self._env_timer_s -= dt
        if self._env_timer_s <= 0:
            self.current_env = random.choice(list(self.ENV_PARAMS))
            self._env_timer_s = random.uniform(400, 1200)
            # Reset stop state on environment change
            self._in_stop = False

    def _city_stop_logic(self, dt: float):
        """Model traffic-light stops in city driving."""
        if self._in_stop:
            self._stop_timer_s -= dt
            if self._stop_timer_s <= 0:
                self._in_stop = False
        else:
            # ~1 stop per 90 s on average when moving slowly
            if self.speed_kmh < 45.0 and random.random() < dt / 90.0:
                self._in_stop = True
                self._stop_duration_s = random.uniform(15.0, 50.0)
                self._stop_timer_s = self._stop_duration_s

    def get_next_step(self, dt: float) -> tuple[float, float]:
        self._maybe_switch_env(dt)

        max_speed, target_speed, theta, sigma = self.ENV_PARAMS[self.current_env]

        # City stop light override
        if self.current_env == 'CITY':
            self._city_stop_logic(dt)
            if self._in_stop:
                target_speed = 0.0

        # OU process: acceleration mean-reverts toward a target
        # μ is a proportional controller on speed error
        speed_error = target_speed - self.speed_kmh
        mu_accel = np.clip(speed_error * 0.05, -2.5, 2.5)  # P-controller

        # OU update: dA = θ(μ - A)dt + σ√dt · N(0,1)
        dW = random.gauss(0.0, 1.0) * np.sqrt(dt)
        self.accel_mps2 += theta * (mu_accel - self.accel_mps2) * dt + sigma * dW

        # Physical acceleration limits (comfort/traction bounds)
        self.accel_mps2 = np.clip(self.accel_mps2, -4.0, 3.5)

        # Integrate to speed
        old_speed = self.speed_kmh
        new_speed = old_speed + self.accel_mps2 * dt * 3.6
        self.speed_kmh = max(0.0, min(max_speed, new_speed))

        # Re-derive exact acceleration from actual speed change
        actual_accel = ((self.speed_kmh - old_speed) / 3.6) / dt

        return self.speed_kmh, actual_accel