"""
Synthetic GPU telemetry dataset generator for NVIDIA H100 SXM5 (700W TDP).

Generates dynamic thermal testing data:
  - Phase 1: Starts cool around 40°C
  - Phase 2: Gradually ramps up to peak heat surge at ~95°C
  - Phase 3: Cools back down and stabilizes in the 65°C - 80°C equilibrium range.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

H100_CONFIG = {
    "max_power":       700.0,
    "min_temp":        30.0,
    "throttle_temp":   90.0,
    "c_thermal":       38.0,
    "air_base":        1.0,
    "air_fan":         2.5,
    "liquid_pump":     16.0,
    "has_liquid":      True,
    "base_gpu_clock":  1000.0,
    "boost_gpu_clock": 1980.0,
    "base_mem_clock":  1000.0,
    "boost_mem_clock": 2619.0,
    "filename":        "h100_telemetry.csv",
}


def make_h100_workload_profile(n: int = 1200):
    """
    Constructs a dynamic 3-phase thermal profile:
      Phase 1: Initial cool baseline (~40°C)
      Phase 2: Gradual heavy ramp up to peak thermal surge (~95°C)
      Phase 3: Cooldown and stabilization into equilibrium (65°C - 80°C)
    """
    t = np.linspace(0, 1, n)
    power_pct = np.zeros(n)
    tensor_util = np.zeros(n)
    ambient_temp = np.zeros(n)
    workload_names = []

    for i, ti in enumerate(t):
        if ti < 0.15:
            # Phase 1: Cool baseline (~40°C start)
            progress = ti / 0.15
            power_pct[i] = 0.18 + 0.05 * progress
            tensor_util[i] = 0.08 + 0.04 * progress
            ambient_temp[i] = 22.0 + 1.0 * progress
            workload_names.append("NOMINAL_COOL_START (40°C)")

        elif ti < 0.60:
            # Phase 2: Gradual ramp from ~40°C up to ~95°C thermal peak
            progress = (ti - 0.15) / 0.45
            power_pct[i] = 0.23 + 0.75 * (progress ** 1.15) + np.random.normal(0, 0.005)
            tensor_util[i] = 0.12 + 0.85 * (progress ** 1.15)
            ambient_temp[i] = 23.0 + 13.5 * (progress ** 1.1)
            workload_names.append("HEAVY_RAMP_UP (-> 95°C Peak Surge)")

        elif ti < 0.75:
            # Sustained peak thermal surge (95°C zone)
            power_pct[i] = 0.98 + np.random.normal(0, 0.005)
            tensor_util[i] = 0.97 + np.random.normal(0, 0.005)
            ambient_temp[i] = 36.5 + 0.5 * np.sin((ti - 0.60) * np.pi / 0.15)
            workload_names.append("THERMAL_PEAK_SURGE (95°C)")

        else:
            # Phase 3: Cooldown and stabilization into 65°C - 80°C range
            progress = (ti - 0.75) / 0.25
            power_pct[i] = 0.98 - 0.42 * progress + np.random.normal(0, 0.008)
            tensor_util[i] = 0.97 - 0.38 * progress
            ambient_temp[i] = 37.0 - 12.0 * progress
            workload_names.append("COOLDOWN_EQUILIBRIUM (65-80°C)")

    power_pct = np.clip(power_pct, 0.15, 1.0)
    tensor_util = np.clip(tensor_util, 0.0, 1.0)
    ambient_temp = np.clip(ambient_temp, 20.0, 40.0)

    return power_pct, tensor_util, ambient_temp, workload_names


def physics_step(temp: float, power: float, fan: float, pump: float, ambient: float, cfg: dict) -> float:
    """Calculates thermal delta using thermodynamic heat transfer."""
    q_in = power * 0.97
    q_out = (temp - ambient) * (cfg["air_base"] + cfg["air_fan"] * fan + cfg["liquid_pump"] * pump * 0.48)
    dT = (q_in - q_out) / cfg["c_thermal"]
    return float(np.clip(temp + dT, ambient, 95.0))


def generate_h100_dataset(n: int = 1200):
    cfg = H100_CONFIG
    power_pct, tensor_util, ambient_temp, workload_names = make_h100_workload_profile(n)

    temps = np.zeros(n)
    fans = np.zeros(n)
    pumps = np.zeros(n)
    powers_w = np.zeros(n)
    gpu_util = np.zeros(n)
    mem_util = np.zeros(n)
    gpu_clocks = np.zeros(n)
    mem_clocks = np.zeros(n)
    power_limits = np.ones(n) * 700.0

    # Explicit initial temperature: 40.0°C
    temps[0] = 40.0

    for i in range(n):
        pw = power_pct[i] * cfg["max_power"]
        powers_w[i] = pw

        t_prev = temps[i - 1] if i > 0 else temps[0]
        ratio = np.clip((t_prev - 30.0) / 60.0, 0.0, 1.0)
        
        # Stock BIOS fan curve
        fan = float(np.clip(0.20 + 0.62 * ratio, 0.20, 0.82))
        pump = float(np.clip(0.20 + 0.62 * ratio, 0.20, 0.82))

        fans[i] = fan
        pumps[i] = pump

        if i > 0:
            temps[i] = physics_step(t_prev, pw, fan, pump, ambient_temp[i], cfg)

        gpu_util[i] = np.clip(power_pct[i] * 0.95 + np.random.normal(0, 0.01), 0.0, 1.0)
        mem_util[i] = np.clip(gpu_util[i] * 0.85 + tensor_util[i] * 0.10 + np.random.normal(0, 0.02), 0.0, 1.0)

        thermal_throttle_factor = 1.0
        if temps[i] > cfg["throttle_temp"]:
            thermal_throttle_factor = max(0.65, 1.0 - (temps[i] - cfg["throttle_temp"]) * 0.04)

        base_c = cfg["base_gpu_clock"]
        boost_c = cfg["boost_gpu_clock"]
        gpu_clocks[i] = (base_c + (boost_c - base_c) * (0.3 + 0.7 * gpu_util[i])) * thermal_throttle_factor

        base_mc = cfg["base_mem_clock"]
        boost_mc = cfg["boost_mem_clock"]
        mem_clocks[i] = (base_mc + (boost_mc - base_mc) * (0.4 + 0.6 * mem_util[i])) * thermal_throttle_factor

    timestamps = pd.date_range("2026-08-10 00:00:00", periods=n, freq="5s")

    df = pd.DataFrame({
        "Timestamp":              timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "GPU_Temp_C":             np.round(temps, 2),
        "GPU_Power_W":            np.round(powers_w, 2),
        "GPU_Utilization_Pct":    np.round(gpu_util * 100, 2),
        "GPU_Clock_MHz":          np.round(gpu_clocks, 1),
        "Memory_Clock_MHz":       np.round(mem_clocks, 1),
        "Memory_Util_Pct":        np.round(mem_util * 100, 2),
        "Tensor_Core_Util_Pct":   np.round(tensor_util * 100, 2),
        "Fan_Speed_Pct":          np.round(fans * 100, 2),
        "Ambient_Temp_C":         np.round(ambient_temp, 2),
        "Power_Limit_W":          np.round(power_limits, 1),
        "Workload_Intensity":     workload_names,
    })

    out = os.path.join(OUTPUT_DIR, cfg["filename"])
    df.to_csv(out, index=False)

    print(f"[OK] Generated Dynamic H100 Telemetry Dataset: {n} rows")
    print(f"   Power: {powers_w.min():.1f}W - {powers_w.max():.1f}W")
    print(f"   Temp Range: {temps.min():.1f}C -> Peak {temps.max():.1f}C -> Settles at {temps[-1]:.1f}C")
    print(f"   Saved to: {out}\n")


if __name__ == "__main__":
    generate_h100_dataset(n=1200)