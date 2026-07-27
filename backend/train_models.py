import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from ml.lstm.model import ThermalLSTM
from ml.rl.environment import GPUCoolingEnv

# ==========================================================
# Create directories for model weights
# ==========================================================
os.makedirs("ml/lstm/weights", exist_ok=True)
os.makedirs("ml/rl/weights", exist_ok=True)

# ==========================================================
# GPU Tier Configuration
# ==========================================================
TIERS = {
    "H100": {"csv": "../data/h100_telemetry.csv", "features": 8},
    "RTX6000": {"csv": "../data/rtx6000_telemetry.csv", "features": 5},
    "RTX4050": {"csv": "../data/rtx4050_telemetry.csv", "features": 4}
}


# ==========================================================
# LSTM Training
# ==========================================================
def train_lstm(
    tier,
    csv_path,
    num_features,
    window_size=40,
    future_steps=5,
    epochs=35
):
    print(f"\n--- Training Tuned LSTM Predictor for Tier: {tier} ---")

    if not os.path.exists(csv_path):
        csv_path = csv_path.replace("../data/", "data/")

    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c != "Timestamp"]
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    Y = []

    for i in range(len(scaled_data) - window_size - future_steps):
        X.append(scaled_data[i:i + window_size])
        Y.append(
            scaled_data[
                i + window_size:
                i + window_size + future_steps,
                :
            ]
        )

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    dataset = TensorDataset(X, Y)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    # Slightly smaller model reduces overfitting
    model = ThermalLSTM(
        input_size=num_features,
        hidden_size=48,
        num_layers=2,
        future_steps=future_steps
    )

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            # Stronger input noise acts as data augmentation
            noise = torch.randn_like(batch_x) * 0.03

            outputs = model(batch_x + noise)

            loss = criterion(outputs, batch_y)

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {total_loss / len(loader):.6f}"
            )

    torch.save(
        model.state_dict(),
        f"ml/lstm/weights/lstm_{tier.lower()}.pt"
    )

    print(
        f"Saved LSTM weights to "
        f"ml/lstm/weights/lstm_{tier.lower()}.pt"
    )


# ==========================================================
# PPO RL Training
# ==========================================================
def train_rl(tier, timesteps=50000):

    print(f"\n--- Training PPO RL Agent for Tier: {tier} ---")

    env = GPUCoolingEnv(tier=tier)

    agent = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2
    )

    agent.learn(total_timesteps=timesteps)

    agent.save(
        f"ml/rl/weights/ppo_{tier.lower()}.zip"
    )

    print(
        f"Saved RL Agent weights to "
        f"ml/rl/weights/ppo_{tier.lower()}.zip"
    )


# ==========================================================
# Main Training Pipeline
# ==========================================================
if __name__ == "__main__":

    for tier, info in TIERS.items():

        train_lstm(
            tier=tier,
            csv_path=info["csv"],
            num_features=info["features"]
        )

        train_rl(tier)

    print("\nTraining Pipeline Complete!")