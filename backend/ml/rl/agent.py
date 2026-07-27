import os
from stable_baselines3 import PPO
from .environment import GPUCoolingEnv

class RLAgent:
    def __init__(self, tier="RTX6000"):
        self.env = GPUCoolingEnv(tier=tier)
        
        weight_path = f"ml/rl/weights/ppo_{tier.lower()}.zip"
        if not os.path.exists(weight_path):
            weight_path = f"backend/ml/rl/weights/ppo_{tier.lower()}.zip"
            
        if os.path.exists(weight_path):
            self.model = PPO.load(weight_path, env=self.env)
            print(f"Loaded PPO RL agent for {tier}")
        else:
            print(f"Warning: No RL weights found for {tier}, initializing untrained PPO.")
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            
    def get_action(self, state_vector):
        action, _ = self.model.predict(state_vector, deterministic=True)
        return action