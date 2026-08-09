"""
Training script for PyTorch Thermal LSTM Predictor and PPO RL Agent for NVIDIA H100.
Saves model weights to ml/lstm/weights/ and ml/rl/weights/.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

from ml.lstm.model import ThermalLSTM
from ml.rl.environment import H100CoolingEnv
from stable_baselines3 import PPO


def train_h100_lstm(data_path: str = "data/h100_telemetry.csv", epochs: int = 20):
    print("\n[INFO] Training PyTorch Thermal LSTM Predictor for H100...")

    if not os.path.exists(data_path):
        data_path = os.path.join("..", data_path)

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "Timestamp" and c != "Workload_Intensity"]

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols].values)

    seq_len = 16
    future_steps = 5

    X, y = [], []
    temp_idx = feature_cols.index("GPU_Temp_C")

    for i in range(len(scaled_data) - seq_len - future_steps + 1):
        X.append(scaled_data[i: i + seq_len])
        y.append(scaled_data[i + seq_len: i + seq_len + future_steps, temp_idx])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ThermalLSTM(
        input_size=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        future_steps=future_steps,
        dropout=0.20
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"   Epoch {ep+1}/{epochs} - Loss: {ep_loss / len(loader):.6f}")

    # Save weights & scaler
    weights_dir = os.path.join(os.path.dirname(__file__), "ml", "lstm", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    weight_path = os.path.join(weights_dir, "lstm_h100.pt")
    scaler_path = os.path.join(weights_dir, "scaler_h100.pkl")

    torch.save(model.state_dict(), weight_path)
    joblib.dump(scaler, scaler_path)

    print(f"[OK] Saved LSTM weights -> {weight_path}")
    print(f"[OK] Saved Scaler -> {scaler_path}")


def train_h100_ppo(total_timesteps: int = 15000):
    print("\n[INFO] Training Stable-Baselines3 PPO RL Agent for H100...")
    env = H100CoolingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0
    )

    model.learn(total_timesteps=total_timesteps)

    weights_dir = os.path.join(os.path.dirname(__file__), "ml", "rl", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    ppo_path = os.path.join(weights_dir, "ppo_h100.zip")
    model.save(ppo_path)

    print(f"[OK] Saved PPO Agent weights -> {ppo_path}")


if __name__ == "__main__":
    train_h100_lstm(epochs=15)
    train_h100_ppo(total_timesteps=15000)
    print("\n[SUCCESS] Training Completed Successfully!")