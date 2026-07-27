import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

def generate_workload_dataset(tier_name, duration_steps=2160, params=None):
    np.random.seed(42)
    time_idx = pd.date_range(start="2024-01-01", periods=duration_steps, freq="5s")
    df = pd.DataFrame(index=time_idx)
    
    utilization = np.zeros(duration_steps)
    power_draw = np.zeros(duration_steps)
    
    max_power = params['max_power']
    current_util = 20.0
    phase_timer = 0
    
    for i in range(duration_steps):
        # 80% Heavy Compute / 20% Idle Profile
        if phase_timer <= 0:
            phase = np.random.choice(['idle', 'heavy'], p=[0.2, 0.8])
            if phase == 'idle':
                phase_timer = np.random.randint(2, 6) # 1 to 2 min idle
                target_util = np.random.uniform(25, 45)
            else:
                phase_timer = np.random.randint(36, 72) # 3 to 6 min compute load
                target_util = np.random.uniform(75, 100)
                
        # Smooth utilization momentum
        current_util += (target_util - current_util) * 0.15
        utilization[i] = np.clip(current_util + np.random.normal(0, 1.0), 0, 100)
        phase_timer -= 1
        
        util_ratio = utilization[i] / 100.0
        target_power = max(max_power * 0.2, util_ratio * max_power)
        power_draw[i] = target_power + np.random.normal(0, max_power * 0.09)

    df['GPU_Util'] = np.clip(utilization, 0, 100)
    df['Power_Draw_W'] = np.clip(power_draw, 10, max_power)
    
    # Initial Baseline Temperature seed for LSTM feature scaling
    df['Die_Temp_C'] = params['min_temp'] + (df['GPU_Util'] / 100.0) * (params['throttle_temp'] - params['min_temp']) * 0.7
    
    if tier_name == "H100":
        df['VRAM_Util'] = df['GPU_Util'] * np.random.uniform(0.8, 1.0)
        df['Core_Clock_MHz'] = np.where(df['Die_Temp_C'] > params['throttle_temp'] - 5, 1200, 1980)
        df['Coolant_Flow_LPM'] = 2.0
        df['Coolant_Inlet_Temp_C'] = 25.0
        df['Ambient_Room_Temp_C'] = 22.0
    elif tier_name == "RTX6000":
        df['Core_Clock_MHz'] = np.where(df['Die_Temp_C'] > params['throttle_temp'] - 5, 1500, 2500)
        df['Fan_Speed_RPM'] = 1500
    elif tier_name == "RTX4050":
        df['Shared_Fan_Speed_RPM'] = 1800

    filename = f"data/{tier_name.lower()}_telemetry.csv"
    df.to_csv(filename, index_label="Timestamp")
    print(f"Generated workload dataset for {tier_name}: {filename}")

tiers = {
    "H100": {"max_power": 700, "min_temp": 60, "throttle_temp": 95},
    "RTX6000": {"max_power": 300, "min_temp": 50, "throttle_temp": 80},
    "RTX4050": {"max_power": 85, "min_temp": 40, "throttle_temp": 70} 
}

for name, params in tiers.items():
    generate_workload_dataset(name, params=params)