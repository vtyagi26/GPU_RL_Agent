"""
RL Agent for NVIDIA H100.
Loads trained PPO policy model and outputs action vectors.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from .environment import H100CoolingEnv


class RLAgent:

    def __init__(self, tier: str = "H100"):
        self.tier = tier
        self.env = H100CoolingEnv()
        self._prev_action = None
        self._alpha = 0.35

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        weight_path = os.path.join(base, "ml", "rl", "weights", f"ppo_{tier.lower()}.zip")

        if os.path.exists(weight_path):
            self.model = PPO.load(weight_path)
            print(f"Loaded PPO weights for {tier}")
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=0)

    def reset(self):
        self._prev_action = None

    def validate_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32).flatten()
        if obs.shape[0] < 16:
            obs = np.pad(obs, (0, 16 - obs.shape[0]))
        elif obs.shape[0] > 16:
            obs = obs[:16]
        return obs

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = self.validate_state(obs)
        raw_action, _ = self.model.predict(obs, deterministic=True)

        if self._prev_action is None:
            smooth = raw_action
        else:
            smooth = self._alpha * raw_action + (1.0 - self._alpha) * self._prev_action

        self._prev_action = smooth
        return smooth.astype(np.float32)