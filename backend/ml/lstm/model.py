import torch
import torch.nn as nn


class ThermalLSTM(nn.Module):
    """
    GPU Thermal Forecasting LSTM

    Input:
        (batch, sequence_length, num_features)

    Output:
        (batch, future_steps)

    Predicts ONLY future GPU temperatures.
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        future_steps=5,
        dropout=0.30
    ):
        super().__init__()

        self.future_steps = future_steps
        self.hidden_size = hidden_size

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

            nn.Linear(hidden_size // 2, future_steps)

        )

        self.initialize_weights()

    # ------------------------------------------------------------

    def initialize_weights(self):

        for name, param in self.named_parameters():

            if "weight_ih" in name:

                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:

                nn.init.orthogonal_(param)

            elif "bias" in name:

                nn.init.constant_(param, 0.)

    # ------------------------------------------------------------

    def forward(self, x):

        """
        x
        (batch, seq_len, features)

        returns

        (batch, future_steps)
        """

        out, _ = self.lstm(x)

        last_hidden = out[:, -1, :]

        last_hidden = self.norm(last_hidden)

        prediction = self.head(last_hidden)

        return prediction