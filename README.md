# GPU Thermal RL Optimizer 🚀🌡️

A full-stack, real-time AI simulation platform that uses a **2-Brain Architecture** (LSTM + Reinforcement Learning) to proactively optimize datacenter and consumer GPU cooling systems. 

By replacing reactive, hardcoded BIOS fan curves with predictive AI, this system minimizes thermal throttling, reduces power consumption, and maximizes hardware longevity across three distinct GPU tiers.

---

## 🧠 The 2-Brain Architecture

The core of this system relies on two decoupled neural networks working in tandem inside a custom first-principles thermodynamic physics engine:

1. **The Forecaster (PyTorch LSTM):** 
   A multi-variate recurrent neural network that ingests real-time hardware telemetry (Power Draw, GPU Utilization, Clock Speeds, etc.) to predict the `Die_Temp_C` 5 seconds into the future. It uses Gaussian noise injection during training to prevent identity-mapping overfits.
2. **The Controller (Stable-Baselines3 PPO Agent):** 
   A Reinforcement Learning agent that receives the hardware state and the LSTM's forward-looking forecast. It adjusts Fan Speed and Liquid Coolant Flow continuously. Its reward function penalizes high temperatures, aggressive mechanical jitter (to save motor life), and wasted power.

## ⚙️ Thermodynamic Physics Engine

The backend features a custom Gymnasium environment (`GPUCoolingEnv`) that calculates temperature deltas using first-principles heat transfer:
* **Heat Generation:** Derived from simulated electrical power (Watts).
* **Heat Dissipation:** Calculated via convection and forced fluid flow (Air + Liquid).
* **Simulation Dual-Tracking:** The backend runs the physics engine *twice* per tick—once using stock BIOS policies (Unoptimized) and once using the RL Agent's policies (Optimized)—allowing for real-time performance comparison.

---

## 🛠️ Tech Stack

**Backend (Machine Learning & Streaming)**
* **Framework:** FastAPI
* **RL Framework:** Stable-Baselines3, Gymnasium
* **Deep Learning:** PyTorch (LSTM)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Streaming:** WebSockets (asyncio)

**Frontend (Visualization)**
* **Framework:** React (Vite)
* **Styling:** Tailwind CSS, Lucide Icons
* **Charts:** Recharts (handling real-time data buffering and asynchronous forecasting)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gpu-thermal-rl-optimizer.git
cd gpu-thermal-rl-optimizer
```

### 2. Backend Setup & Model Training
Navigate to the backend, set up your virtual environment, and generate the physical datasets.

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: . env\Scripts ctivate
pip install -r requirements.txt

# 1. Generate 80/20 sustained workload datasets for all GPU tiers
python generate_datasets.py

# 2. Train the PyTorch LSTM & PPO RL Agent (This will save weights to ml/lstm/weights and ml/rl/weights)
python train_models.py
```

### 3. Start the FastAPI WebSocket Server
Once training is complete, spin up the backend stream.
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Start the React Frontend
Open a new terminal window, navigate to the frontend directory, and start the UI.
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` (or your Vite local port) to view the live dashboard!

---

## 🖥️ Supported Hardware Tiers

The system simulates physics and controls specifically tuned for three tiers of hardware:

| Tier | Cooling Method | Throttle Limit | Peak Power | Action Space |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA H100** | Liquid + Air | 110°C | 700W | `[Pump %, Fan %]` |
| **RTX 6000 Ada** | Air (Blower) | 90°C | 300W | `[Fan %]` |
| **RTX 4050 Mobile**| Air (Shared) | 85°C | 85W | `[Fan %]` |

---

## 📊 Dashboard Features
* **Independent Forecasting:** The yellow LSTM prediction line is plotted natively in the future (T+5s), completely decoupled from the current temperature anchor.
* **Live Delta Tracking:** Real-time visibility into the exact ±% RPM/LPM differential between the stock BIOS curve and the RL Agent's requested policy.
* **Resource Savings:** Continuous accumulation of watts and coolant liters saved by the RL agent's efficiency.