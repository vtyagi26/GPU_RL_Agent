import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GPUCoolingEnv(gym.Env):
    def __init__(self, tier="RTX6000"):
        super(GPUCoolingEnv, self).__init__()
        self.tier = tier
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
        
        if tier == "H100":
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) 
            self.min_temp, self.throttle_temp, self.target_temp = 60.0, 110.0, 85.0
            self.c_thermal = 25.0 
        elif tier == "RTX6000":
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) 
            self.min_temp, self.throttle_temp, self.target_temp = 50.0, 90.0, 75.0
            self.c_thermal = 15.0
        else: 
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) 
            self.min_temp, self.throttle_temp, self.target_temp = 40.0, 85.0, 65.0
            self.c_thermal = 10.0
            
        self.state = np.zeros(13, dtype=np.float32)
        self.temp = self.min_temp + 5.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = self.min_temp + 5.0
        self.state = np.random.uniform(0.2, 0.8, size=(13,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        phys_action = np.clip((action + 1.0) / 2.0, 0.35, 1.0) # 25% to 100% fan speed
        
        fan_pct = phys_action[0] if self.tier != "H100" else phys_action[1]
        coolant_pct = phys_action[0] if self.tier == "H100" else 0.0
        
        # Heavy Compute Simulation (300 Watts)
        ambient_temp = 30.0
        q_in = 280.0 
        
        # Fan/Coolant directly rip heat out of the system
        air_cooling = (self.temp - ambient_temp) * (0.5 + 4.0 * fan_pct)
        liquid_cooling = (self.temp - 40.0) * (6.0 * coolant_pct) if self.tier == "H100" else 0.0
        
        q_out = air_cooling + liquid_cooling
        
        d_temp = (q_in - q_out) / self.c_thermal
        self.temp = np.clip(self.temp + d_temp, self.min_temp, self.throttle_temp)
        
        # --- NEW AGGRESSIVE REWARD POLICY ---
        reward = 0.0
        
        if self.temp > self.target_temp:
            # Massive penalty for being hot
            reward -= (self.temp - self.target_temp) * 1.7
            
            # Massive REWARD for cranking fans when hot (This forces the RL agent to lower temps!)
            if np.mean(phys_action) > 0.8:
                reward += 20.0 
        else:
            # Reward for staying in the cold zone
            reward += 10.0
            # Small penalty for wasting fans when it's already cold
            if np.mean(phys_action) > 0.5:
                reward -= 55.0
                
        if self.temp >= self.throttle_temp - 2.0:
            reward -= 80.0  
            
        self.state[0] = (self.temp - self.min_temp) / (self.throttle_temp - self.min_temp)
        
        terminated = bool(self.temp >= self.throttle_temp)
        return self.state.astype(np.float32), float(reward), terminated, False, {}