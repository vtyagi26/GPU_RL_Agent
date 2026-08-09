"""
NVIDIA H100 Gym Environment (10 Parameters + Thermal Velocity).
Observation space: 16 floats.
Action space: 2 floats [-1, 1] -> [Cooling Level (Fan/Pump %), Power Limit Constraint (W)].
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

H100_ENV_CONFIG = {
    "max_power":       700.0,
    "min_power":       300.0,
    "min_temp":        30.0,
    "throttle_temp":   90.0,
    "c_thermal":       32.0,
    "max_fan_power_w": 35.0,
    "air_base":        1.2,
    "air_fan":         3.0,
    "liquid_pump":     24.0,
}


class H100CoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        cfg = H100_ENV_CONFIG
        self.max_power = cfg["max_power"]
        self.min_temp = cfg["min_temp"]
        self.throttle_temp = cfg["throttle_temp"]
        self.c_thermal = cfg["c_thermal"]
        self.air_base = cfg["air_base"]
        self.air_fan = cfg["air_fan"]
        self.liquid_pump = cfg["liquid_pump"]

        # 16-dim observation space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

        # 2-dim action space: [Cooling Level %, Power Limit Cap W]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.temp = self.min_temp + 10.0
        self.unopt_temp = self.temp + 2.0
        self.power = 400.0
        self.ambient = 24.0
        self.current_fan = 0.40
        self.current_power_limit = 700.0
        self.step_count = 0
        self.prev_temp = self.temp

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = self.min_temp + np.random.uniform(5, 20)
        self.unopt_temp = self.temp + np.random.uniform(0, 3)
        self.power = np.random.uniform(200.0, 650.0)
        self.ambient = np.random.uniform(20.0, 32.0)
        self.current_fan = 0.40
        self.current_power_limit = 700.0
        self.step_count = 0
        self.prev_temp = self.temp

        obs = np.random.uniform(0.1, 0.5, 16).astype(np.float32)
        return obs, {}

    def _physics(self, temp: float, power: float, fan: float, pump: float, ambient: float) -> float:
        q_in = min(power, self.current_power_limit) * 0.96
        q_out = (temp - ambient) * (self.air_base + self.air_fan * fan + self.liquid_pump * pump)
        dT = (q_in - q_out) / self.c_thermal
        return float(np.clip(temp + dT, ambient, 110.0))

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # Action 0: Cooling duty [0.20, 1.00]
        fan = float(0.20 + 0.80 * (action[0] + 1.0) / 2.0)
        # Action 1: Power Limit [300W, 700W]
        power_limit = float(300.0 + 400.0 * (action[1] + 1.0) / 2.0)

        self.current_fan = fan
        self.current_power_limit = power_limit

        # Stochastic power fluctuation
        self.power += np.random.normal(0, 20.0)
        self.power = float(np.clip(self.power, 100.0, 700.0))

        # Update temperatures
        self.prev_temp = self.temp
        effective_power = min(self.power, self.current_power_limit)
        self.temp = self._physics(self.temp, effective_power, fan, fan, self.ambient)

        # Stock BIOS step
        stock_fan = float(np.clip(0.20 + 0.78 * ((self.unopt_temp - 30.0) / 60.0), 0.20, 1.0))
        self.unopt_temp = self._physics(self.unopt_temp, self.power, stock_fan, stock_fan, self.ambient)

        dt_dt = (self.temp - self.prev_temp) / 5.0

        # Reward formulation
        reward = 5.0
        # Hard constraint: Never exceed BIOS unopt temp by > 0.5C
        if self.temp > self.unopt_temp + 0.5:
            reward -= (self.temp - self.unopt_temp) * 40.0

        # Hard constraint: Never exceed 90C throttle junction
        if self.temp > 88.0:
            reward -= (self.temp - 88.0) * 80.0

        # Reward maintaining safe temperature headroom with minimal cooling energy
        if 60.0 <= self.temp <= 78.0:
            reward += 20.0
            # Conserve fan power when safe
            reward += (stock_fan - fan) * 30.0

        self.step_count += 1
        done = self.step_count >= 1000

        # Build 16-dim observation
        obs = np.zeros(16, dtype=np.float32)
        obs[0] = np.clip((self.temp - 30.0) / 60.0, 0, 1)
        obs[1] = np.clip(effective_power / 700.0, 0, 1)
        obs[2] = obs[1] * 0.95
        obs[3:8] = obs[0]  # Future temps proxy
        obs[8] = fan
        obs[9] = fan
        obs[10] = np.clip((self.ambient - 18.0) / 22.0, 0, 1)
        obs[11] = obs[2]
        obs[12] = np.clip((self.current_power_limit - 300.0) / 400.0, 0, 1)
        obs[13] = np.clip(dt_dt / 2.0, -1, 1)
        obs[14] = 0.0
        obs[15] = 0.0

        return obs, reward, done, False, {}