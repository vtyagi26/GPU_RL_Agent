"""
LSTM Predictor — Fixed calibration & sequence padding.

Ensures:
  1. Sequence padding so LSTM predicts accurate future Stock BIOS temperatures right from step 0.
  2. Direct mathematical inverse transform using scaler bounds (data_min_ & data_max_).
  3. Strict alignment with future Stock BIOS thermal trajectory.
"""

import os
import numpy as np
import torch
import joblib

from .model import ThermalLSTM


class LSTMPredictor:

    SEQUENCE_LENGTH = 16   # 16 × 5s = 80s of history for fast responsive forecasting
    FUTURE_STEPS    = 5

    def __init__(self, num_features: int, tier: str = "h100"):
        self.tier         = tier.lower()
        self.num_features = num_features
        self.history      = []
        self.scaler       = None
        self.temp_idx     = 0  # GPU_Temp_C is column index 0

        self.model = ThermalLSTM(
            input_size   = num_features,
            hidden_size  = 64,
            num_layers   = 2,
            future_steps = self.FUTURE_STEPS,
            dropout      = 0.20,
        )
        self.model.eval()

        # Resolve weight paths
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._weight_path = self._resolve(base, f"ml/lstm/weights/lstm_{self.tier}.pt")
        self._scaler_path = self._resolve(base, f"ml/lstm/weights/scaler_{self.tier}.pkl")

        if os.path.exists(self._weight_path):
            self.model.load_state_dict(
                torch.load(self._weight_path, map_location="cpu", weights_only=True)
            )
            print(f"[OK] Loaded PyTorch LSTM weights for {tier}")
        else:
            print(f"[WARNING] LSTM weights not found for {tier}. Using initialized model.")

        if os.path.exists(self._scaler_path):
            self.scaler = joblib.load(self._scaler_path)
            print(f"[OK] Loaded MinMaxScaler for {tier}")
        else:
            print(f"[WARNING] MinMaxScaler not found for {tier}.")

    def _resolve(self, base, rel):
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            p = rel
        return p

    def reset_history(self):
        self.history = []

    def update_history(self, row: np.ndarray):
        """Push one telemetry row (raw, unscaled)."""
        self.history.append(row.astype(np.float32))
        if len(self.history) > self.SEQUENCE_LENGTH:
            self.history = self.history[-self.SEQUENCE_LENGTH:]

    def predict(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Pushes current telemetry row, returns array of 5 future Stock BIOS temperatures (°C).
        Pads initial sequence so forecast is dynamic right from step 0.
        """
        self.update_history(telemetry)

        # Pad sequence with initial row if history is not full yet
        if len(self.history) < self.SEQUENCE_LENGTH:
            pad_count = self.SEQUENCE_LENGTH - len(self.history)
            padded_seq = [self.history[0]] * pad_count + self.history
        else:
            padded_seq = self.history

        seq = np.array(padded_seq, dtype=np.float32)  # (16, num_features)

        # Apply MinMaxScaler transform
        if self.scaler is not None:
            seq_scaled = self.scaler.transform(seq)
        else:
            col_max = np.abs(seq).max(axis=0) + 1e-8
            seq_scaled = seq / col_max

        tensor = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            pred_norm = self.model(tensor).numpy()[0]  # (5,) normalized

        # Inverse Transform for GPU_Temp_C (column 0)
        if self.scaler is not None and hasattr(self.scaler, "data_min_") and hasattr(self.scaler, "data_max_"):
            min_temp = float(self.scaler.data_min_[self.temp_idx])
            max_temp = float(self.scaler.data_max_[self.temp_idx])
            future_temps = pred_norm * (max_temp - min_temp) + min_temp
        else:
            current_temp = float(telemetry[self.temp_idx])
            future_temps = current_temp + pred_norm * 15.0

        # Sanity bounds check
        future_temps = np.clip(future_temps, 30.0, 100.0)
        return future_temps.astype(np.float32)