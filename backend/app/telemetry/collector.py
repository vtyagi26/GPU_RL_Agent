"""
Telemetry Collector Component.
Parses live GPU telemetry from nvidia-smi / DCGM or replays synthetic H100 telemetry.
"""

import os
import subprocess
import pandas as pd
import numpy as np


class TelemetryCollector:

    def __init__(self, data_path: str = "data/h100_telemetry.csv"):
        self.data_path = data_path
        self.df = None
        self.use_live_nvidia_smi = False
        self._load_dataset()

    def _load_dataset(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            # Fallback path if invoked from backend/ directory
            alt_path = os.path.join("..", self.data_path)
            if os.path.exists(alt_path):
                self.df = pd.read_csv(alt_path)
            else:
                raise FileNotFoundError(f"Telemetry dataset not found at {self.data_path} or {alt_path}")

    def query_live_nvidia_smi(self) -> dict | None:
        """Attempts to query live nvidia-smi if an NVIDIA GPU is physically present."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                return {
                    "GPU_Temp_C": float(parts[0]),
                    "GPU_Power_W": float(parts[1]),
                    "GPU_Utilization_Pct": float(parts[2]),
                    "Memory_Util_Pct": float(parts[3]),
                    "GPU_Clock_MHz": float(parts[4]),
                    "Memory_Clock_MHz": float(parts[5]),
                    "Tensor_Core_Util_Pct": float(parts[2]) * 0.9,
                    "Fan_Speed_Pct": 55.0,
                    "Ambient_Temp_C": 24.0,
                    "Power_Limit_W": 700.0,
                    "Workload_Intensity": "LIVE_GPU_NODE",
                    "Source": "nvidia-smi"
                }
        except Exception:
            pass
        return None

    def read_step(self, step_idx: int) -> dict:
        """Reads telemetry for a given step index (cycling dataset)."""
        idx = step_idx % len(self.df)
        row = self.df.iloc[idx].to_dict()
        row["Step"] = step_idx
        row["Source"] = "H100_Telemetry_Stream"
        return row
