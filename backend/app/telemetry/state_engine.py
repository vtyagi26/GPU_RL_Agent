"""
Thermal State Engine.
Computes rich thermal physics state:
  - Current Temperature (°C)
  - Temperature Velocity (dT/dt in °C/sec)
  - Power Acceleration Trend (dP/dt in W/sec)
  - Tensor / SM Workload Trend (dW/dt)
  - Predicted Temperatures (T+1 ... T+N)
"""

from collections import deque
import numpy as np


class ThermalStateEngine:

    def __init__(self, history_len: int = 5):
        self.history_len = history_len
        self.temp_history = deque(maxlen=history_len)
        self.power_history = deque(maxlen=history_len)
        self.tensor_history = deque(maxlen=history_len)

    def reset(self):
        self.temp_history.clear()
        self.power_history.clear()
        self.tensor_history.clear()

    def update(self, telemetry: dict, predicted_temps: list[float]) -> dict:
        """
        Ingests telemetry row and predictions to construct the full Thermal State.
        """
        temp = float(telemetry.get("GPU_Temp_C", 35.0))
        power = float(telemetry.get("GPU_Power_W", 200.0))
        tensor = float(telemetry.get("Tensor_Core_Util_Pct", 0.0))
        ambient = float(telemetry.get("Ambient_Temp_C", 24.0))

        self.temp_history.append(temp)
        self.power_history.append(power)
        self.tensor_history.append(tensor)

        # ── Temperature Velocity (dT/dt) ──
        if len(self.temp_history) >= 2:
            # 5s step interval in simulation
            dt_dt = round((self.temp_history[-1] - self.temp_history[-2]) / 5.0, 2)
        else:
            dt_dt = 0.0

        # ── Power Acceleration Trend (dP/dt) ──
        if len(self.power_history) >= 2:
            dp_dt = round((self.power_history[-1] - self.power_history[-2]) / 5.0, 2)
        else:
            dp_dt = 0.0

        # ── Workload Trend (dW/dt) ──
        if len(self.tensor_history) >= 2:
            dw_dt = round((self.tensor_history[-1] - self.tensor_history[-2]) / 5.0, 2)
        else:
            dw_dt = 0.0

        # Thermal Headroom to Throttle Limit (90°C for H100)
        throttle_limit = 90.0
        thermal_headroom = round(throttle_limit - temp, 2)

        # Max predicted temp over T+1 ... T+N
        max_predicted_temp = max(predicted_temps) if predicted_temps else temp

        # Predictor trend delta (T+N predicted minus current temp)
        pred_delta_t = round(max_predicted_temp - temp, 2)

        return {
            "current_temp":          temp,
            "temp_velocity_dt_dt":   dt_dt,
            "power_trend_dp_dt":     dp_dt,
            "workload_trend_dw_dt":  dw_dt,
            "ambient_temp":          ambient,
            "power_draw_w":          power,
            "gpu_util_pct":          float(telemetry.get("GPU_Utilization_Pct", 0.0)),
            "tensor_core_util_pct":  tensor,
            "gpu_clock_mhz":         float(telemetry.get("GPU_Clock_MHz", 1000.0)),
            "memory_clock_mhz":      float(telemetry.get("Memory_Clock_MHz", 1000.0)),
            "memory_util_pct":       float(telemetry.get("Memory_Util_Pct", 0.0)),
            "thermal_headroom_c":    thermal_headroom,
            "predicted_temps":       predicted_temps,
            "max_predicted_temp":    round(max_predicted_temp, 2),
            "predictor_trend_delta": pred_delta_t,
            "workload_intensity":    str(telemetry.get("Workload_Intensity", "IDLE")),
        }
