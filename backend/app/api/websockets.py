"""
WebSocket Endpoint for Real-Time NVIDIA H100 Telemetry & Agentic AI Stream.
"""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.simulation import SimulationEngine
from app.api.routes import active_policy
import os

ws_router = APIRouter()


@ws_router.websocket("/ws/telemetry/h100")
@ws_router.websocket("/ws/telemetry/{tier}")
async def websocket_telemetry(websocket: WebSocket, tier: str = "H100"):
    await websocket.accept()

    data_filename = "data/h100_telemetry.csv"
    if not os.path.exists(data_filename):
        data_filename = "../data/h100_telemetry.csv"

    try:
        engine = SimulationEngine(tier="H100", data_path=data_filename)
        current_policy = active_policy["mode"]
        engine.set_agent_policy(current_policy)

        while True:
            # Sync policy if updated via REST
            if active_policy["mode"] != current_policy:
                current_policy = active_policy["mode"]
                engine.set_agent_policy(current_policy)

            payload = await engine.run_step()
            await websocket.send_text(json.dumps(payload))

            # Stream 1 payload per second (representing 5s simulation step)
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        print("Client disconnected from H100 telemetry stream")
    except Exception as e:
        print(f"WebSocket Error: {e}")