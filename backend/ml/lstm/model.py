import torch
import torch.nn as nn


class ThermalLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=2,
        future_steps=5,
        dropout=0.4
    ):
        super().__init__()

        self.future_steps = future_steps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, input_size)
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        # Last 5 hidden states
        out = out[:, -self.future_steps:, :]

        # Layer Normalization
        out = self.norm(out)

        # Residual connection
        residual = out

        out = self.head(out)

        # Small residual projection if dimensions match
        if residual.shape[-1] == out.shape[-1]:
            out = out + 0.1 * residual

        return out