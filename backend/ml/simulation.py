import pandas as pd
import numpy as np
from .lstm.predictor import LSTMPredictor
from .rl.agent import RLAgent

class SimulationEngine:
    def __init__(self, tier="RTX6000", data_path="../data/rtx6000_telemetry.csv"):
        self.tier = tier
        self.df = pd.read_csv(data_path)
        self.features = [col for col in self.df.columns if col != 'Timestamp']
        
        self.lstm = LSTMPredictor(num_features=len(self.features), tier=tier)
        self.lstm.update_scaler(self.df[self.features])
        self.agent = RLAgent(tier=tier)
        
        self.current_step = 0
        self.current_rl_actions = None
        
        if tier == "H100":
            self.min_temp, self.throttle_temp = 60.0, 110.0
            self.c_thermal = 25.0
        elif tier == "RTX6000":
            self.min_temp, self.throttle_temp = 50.0, 90.0
            self.c_thermal = 15.0
        else:
            self.min_temp, self.throttle_temp = 40.0, 85.0
            self.c_thermal = 10.0
            
        self.unopt_temp = self.min_temp + 5.0
        self.opt_temp = self.min_temp + 5.0
        
        self.total_power_saved_w = 0.0
        self.total_coolant_saved_l = 0.0

    def calculate_physics(self, current_temp, power_w, fan_pct, coolant_pct=0.0):
        """Unified Heat Transfer Formula"""
        ambient_temp = 22.0
        q_in = power_w * 0.95
        
        # High fan_pct drastically increases q_out, cooling the GPU rapidly
        air_cooling = (current_temp - ambient_temp) * (0.5 + 4.0 * fan_pct)
        liquid_cooling = (current_temp - 25.0) * (6.0 * coolant_pct) if self.tier == "H100" else 0.0
        
        q_out = air_cooling + liquid_cooling
        d_temp = (q_in - q_out) / self.c_thermal
        
        return float(np.clip(current_temp + d_temp, self.min_temp, self.throttle_temp))

    async def run_step(self):
        if self.current_step >= len(self.df):
            self.current_step = 0
            
        row = self.df.iloc[self.current_step]
        current_state = row[self.features].values
        power_w = float(row['Power_Draw_W'])
        
        # --- 1. UNOPTIMIZED THERMODYNAMICS (Lazy Stock BIOS) ---
        # The stock curve only hits 100% when it's basically on fire.
        unopt_fan_pct = min(1.0, max(0.25, (self.unopt_temp - self.min_temp - 10) / (self.throttle_temp - self.min_temp)))
        unopt_coolant_pct = min(1.0, max(0.15, (self.unopt_temp - self.min_temp - 15) / (self.throttle_temp - self.min_temp))) if self.tier == "H100" else 0.0
        
        self.unopt_temp = self.calculate_physics(self.unopt_temp, power_w, unopt_fan_pct, unopt_coolant_pct)
        
        # --- 2. LSTM FORECAST (Fixed Indexing) ---
        predictions = self.lstm.predict(current_state)
        temp_idx = self.features.index("Die_Temp_C")
        temp_min = self.lstm.scaler.data_min_[temp_idx]
        temp_range = self.lstm.scaler.data_max_[temp_idx] - temp_min
        
        # We successfully extract exactly the temperature prediction column!
        lstm_forecast = float((predictions[0, temp_idx] * temp_range) + temp_min)
        
        # --- 3. OPTIMIZED THERMODYNAMICS (RL Agent Action) ---
        flat_predictions = predictions.flatten()
        obs = np.concatenate([[current_state[0]], [current_state[1]], [current_state[2]], flat_predictions])
        if len(obs) < 13: obs = np.pad(obs, (0, 13 - len(obs)))
        else: obs = obs[:13]
            
        raw_action = self.agent.get_action(obs.astype(np.float32))
        target_actions = np.clip((raw_action + 1.0) / 2.0, 0.25, 1.0)
        
        if self.current_rl_actions is None:
            self.current_rl_actions = target_actions
        else:
            delta = np.clip(target_actions - self.current_rl_actions, -0.05, 0.05)
            self.current_rl_actions += delta
            
        opt_fan_pct = float(self.current_rl_actions[0] if self.tier != "H100" else self.current_rl_actions[1])
        opt_coolant_pct = float(self.current_rl_actions[0]) if self.tier == "H100" else 0.0
        
        self.opt_temp = self.calculate_physics(self.opt_temp, power_w, opt_fan_pct, opt_coolant_pct)
        
        # --- 4. EXPORTS ---
        payload = {
            "timestamp": row['Timestamp'],
            "actual_temp": self.unopt_temp,
            "rl_controlled_temp": self.opt_temp,
            "predicted_temp_t5": lstm_forecast,
            "controls": {
                "fan": {
                    "unoptimized_pct": float(unopt_fan_pct * 100),
                    "optimized_pct": float(opt_fan_pct * 100),
                    "delta_pct": float((opt_fan_pct - unopt_fan_pct) * 100)
                },
                "coolant": {
                    "unoptimized_pct": float(unopt_coolant_pct * 100),
                    "optimized_pct": float(opt_coolant_pct * 100),
                    "delta_pct": float((opt_coolant_pct - unopt_coolant_pct) * 100)
                } if self.tier == "H100" else None
            },
            "metrics": {
                "temp_difference": float(self.unopt_temp - self.opt_temp),
                "total_power_saved": 0.0, # Removed to prevent confusion, focus is on thermal cooling
                "total_coolant_saved": 0.0,
            }
        }
        
        self.current_step += 1
        return payload