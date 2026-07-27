import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.simulation import SimulationEngine
import os

ws_router = APIRouter()

@ws_router.websocket("/ws/telemetry/{tier}")
async def websocket_telemetry(websocket: WebSocket, tier: str):
    await websocket.accept()
    tier_clean = tier.upper()
    
    data_filename = f"data/{tier_clean.lower()}_telemetry.csv"
    if not os.path.exists(data_filename):
        data_filename = f"../data/{tier_clean.lower()}_telemetry.csv"
        
    try:
        engine = SimulationEngine(tier=tier_clean, data_path=data_filename)
        while True:
            payload = await engine.run_step()
            await websocket.send_text(json.dumps(payload))
            # Sends data 1 real-world second at a time (representing 5 simulated seconds)
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        print(f"Client disconnected from {tier} telemetry")
    except Exception as e:
        print(f"WebSocket Error on {tier}: {e}")