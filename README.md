# NVIDIA H100 Agentic AI Thermal Controller & RAG System 🚀🌡️🤖

An enterprise-grade, real-time **GenAI & Agentic AI Thermal Optimization Platform** engineered specifically for the **NVIDIA H100 SXM5 80GB (700W TDP)** GPU architecture. 

By replacing reactive, hardcoded BIOS fan curves with a **5-Stage Closed-Loop Agentic AI Controller** (Observe → Reason → Plan → Act → Verify), a **PyTorch Multi-Step LSTM Predictor**, and a **Vector Retrieval-Augmented Generation (RAG) Engine**, this system proactively mitigates thermal throttling, prevents junction thermal runaway, reduces energy consumption, and extends hardware lifespan.

---

## 🏛️ System Architecture

```
                 ┌──────────────────────────┐
                 │       H100 GPU           │
                 │                          │
                 │ Utilization              │
                 │ Power                    │
                 │ Temperature              │
                 │ SM / Tensor utilization  │
                 │ Memory / HBM metrics     │
                 │ Clock                    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Telemetry Collector    │
                 │      nvidia-smi / DCGM   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      Thermal State       │
                 │                          │
                 │ Current temp             │
                 │ Temp velocity            │
                 │ Power trend              │
                 │ Workload trend           │
                 │ Predicted temperature    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Thermal Predictor    │
                 │         LSTM             │
                 │                          │
                 │ Predict T+1 ... T+N      │
                 └────────────┬─────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │     AGENTIC AI          │
                  │   Thermal Controller    │
                  │                         │
                  │ Observe → Reason → Plan │
                  │ → Act → Verify          │
                  └────────────┬────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        Increase fan      Reduce power      Maintain
        cooling           limit             settings
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                         H100 GPU
```

---

## 📊 Monitored & Controlled Parameters

The system ingests and optimizes **10 core NVIDIA H100 parameters**:

| Parameter | Effect on Temperature | Role in System |
| :--- | :--- | :--- |
| **GPU Utilization (%)** | ↑ utilization → ↑ compute activity → ↑ power → ↑ temp | **Primary Input** |
| **GPU Power Usage (W)** | ↑ power → ↑ heat generation | **Primary Input / Target** |
| **GPU Clock / SM Clock (MHz)** | ↑ clock generally → ↑ power/heat | **Important Input / Advisory** |
| **Memory Clock (MHz)** | ↑ HBM3 memory activity/clock → ↑ power | **Important Input** |
| **GPU Memory Utilization (%)** | HBM3 memory traffic power contribution | **Secondary Input** |
| **SM / Tensor Core Utilization (%)** | Heavy FP8/FP16 AI matrix workloads raise power significantly | **Key H100 Workload Indicator** |
| **Fan Speed / Cooling Level (%)** | ↑ cooling → ↓ temperature | **Control Variable (Action 1)** |
| **Ambient / Inlet Temp (°C)** | ↑ ambient → ↑ GPU temperature | **Environmental Constraint** |
| **Power Limit (W)** | Restricts maximum GPU power (300W–700W) | **Control Constraint (Action 2)** |
| **Workload Intensity / Trend** | Trajectory of AI compute load | **Derived State ($dW/dt$)** |

---

## 🔑 Key Features

### 1. 🤖 5-Stage Agentic AI Control Loop
The controller (`backend/ml/agentic/thermal_agent.py`) executes an autonomous 5-stage loop every telemetry tick:
- **1. Observe**: Ingests 10-parameter telemetry, thermal velocity ($dT/dt$), ambient temperature, and LSTM $T+5$ predictions.
- **2. Reason**: Evaluates junction thermal headroom to the **90°C TJunction Throttle Threshold** and assesses ambient limits.
- **3. Plan**: Formulates preemption & dynamic TDP cap reduction strategies.
- **4. Act**: Dispatches discrete commands (`INCREASE_FAN_COOLING`, `REDUCE_POWER_LIMIT`, `MAINTAIN_SETTINGS`).
- **5. Verify**: Performs closed-loop verification check on thermal deceleration.

### 2. 📚 Vector RAG Diagnostic Engine & Copilot Chat
- **Knowledge Base**: Contains official NVIDIA H100 SXM5 technical specs, HBM3 memory thresholds, liquid cooling flow equations, and emergency playbooks.
- **Vector Search Index**: Uses TF-IDF tokenization and Cosine Similarity vector matching (`backend/app/rag/vector_store.py`).
- **Real-Time Telemetry RAG Diagnosis**: Generates executive diagnostic reading reports based on live thermal dynamics.
- **RAG Copilot Chat**: Interactive chat assistant that answers user questions with direct document citations.

### 3. 🧠 PyTorch LSTM Thermal Predictor
- Predicts multi-step ahead die temperatures ($T+1 \dots T+5$, 25 seconds ahead).
- Calibrated to project future Stock BIOS unoptimized temperature trajectories so the AI Agent can act *before* heat spikes occur.

### 4. 📈 Dynamic Testing Profile (40°C → 95°C → 65°C–80°C)
- Replays a dynamic 3-phase testing profile showcasing:
  - **Phase 1**: Cool nominal baseline (~40°C).
  - **Phase 2**: Heavy workload & ambient heatwave surge up to **95°C** (triggering AI emergency preemption & TDP power limit capping).
  - **Phase 3**: Cooldown and stabilization into the safe **65°C – 80°C** equilibrium range.

### 5. 🛡️ Arrhenius Hardware Lifespan & Sustainability HUD
- Real-time counters tracking accrued energy saved (kJ), liquid coolant conserved (Liters), and Arrhenius semiconductor mean time between failures (MTBF) lifespan extension (+Hours / %).

---

## 🛠️ Tech Stack

**Backend (Machine Learning, RAG & Streaming)**
- **Framework:** FastAPI
- **Agentic AI & RL:** Stable-Baselines3 (PPO Agent), Gymnasium (`H100CoolingEnv`)
- **Deep Learning:** PyTorch (LSTM Thermal Predictor)
- **Vector RAG:** Custom Cosine Similarity & TF-IDF Vector Index Engine
- **Data & Math:** Pandas, NumPy, Scikit-Learn, Joblib
- **Streaming:** WebSockets (`asyncio`)

**Frontend (Dashboard & Visualization)**
- **Framework:** React 19 (Vite)
- **Styling:** Vanilla CSS3 (Glassmorphism & Sleek Dark Mode), Lucide Icons
- **Charts:** Recharts (handling real-time data buffering and asynchronous forecasting)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gpu-thermal-rl-optimizer.git
cd gpu-thermal-rl-optimizer
```

### 2. Backend Setup & Dataset Generation
Navigate to the backend, set up your Python environment, and generate the physical telemetry dataset:

```bash
# 1. Generate dynamic H100 10-parameter telemetry dataset
python generate_datasets.py

# 2. Train the PyTorch LSTM & PPO RL Agent (Saves weights to ml/lstm/weights and ml/rl/weights)
python backend/train_models.py
```

### 3. Start the FastAPI Server
Spin up the backend REST API & WebSocket stream:
```bash
cd backend
python -m uvicorn app.main:app --port 8000 --reload
```

### 4. Start the React Frontend Dashboard
Open a new terminal window, navigate to the frontend directory, and start the UI:
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to launch the live **NVIDIA H100 Agentic AI Thermal Control Room**!

---

## 📖 System Documentation & Interview Prep Guide

For detailed technical explanations, mathematical formulations, LangChain/LangGraph concepts, and a 22-question technical interview guide, refer to:
- **[System Documentation & Interview Prep Guide](system_documentation_and_interview_prep.md)**
- **[Implementation Walkthrough](walkthrough.md)**

---

## 📄 License
This project is open-source under the MIT License.