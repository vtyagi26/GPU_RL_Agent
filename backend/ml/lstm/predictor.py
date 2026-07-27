import os
import torch
from sklearn.preprocessing import MinMaxScaler
from .model import ThermalLSTM


class LSTMPredictor:
    def __init__(self, num_features, tier="RTX6000"):
        self.tier = tier

        # Must match training configuration
        self.model = ThermalLSTM(
            input_size=num_features,
            hidden_size=48,
            num_layers=2,
            future_steps=5,
            dropout=0.4
        )

        self.model.eval()
        self.scaler = None

        weight_path = f"ml/lstm/weights/lstm_{tier.lower()}.pt"

        if not os.path.exists(weight_path):
            weight_path = f"backend/ml/lstm/weights/lstm_{tier.lower()}.pt"

        if os.path.exists(weight_path):
            self.model.load_state_dict(
                torch.load(weight_path, map_location=torch.device("cpu"))
            )
            print(f"Loaded LSTM weights for {tier}")
        else:
            print(f"Warning: No LSTM weights found for {tier}")

    def update_scaler(self, df):
        """
        Fit scaler using telemetry dataframe.
        Should be called once before prediction.
        """
        self.scaler = MinMaxScaler()
        self.scaler.fit(df.values)

    def predict(self, current_state):
        """
        Predict the next 5 timesteps.

        Args:
            current_state: list/array of current telemetry values.

        Returns:
            numpy.ndarray of shape (5, num_features)
        """

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been initialized. "
                "Call update_scaler() before predict()."
            )

        # Scale current telemetry
        scaled = self.scaler.transform([current_state])

        # Create a sequence of length 40 (must match training window_size)
        seq = (
            torch.tensor(scaled, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(1, 40, 1)
        )

        with torch.no_grad():
            prediction = self.model(seq)

        return prediction.numpy()[0]