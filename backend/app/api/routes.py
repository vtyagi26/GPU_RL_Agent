"""
REST API Routes for NVIDIA H100 Agentic AI Thermal Controller & RAG Engine.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag.rag_engine import RAGEngine
from app.rag.knowledge_base import H100_KNOWLEDGE_DOCUMENTS

router = APIRouter()
rag_engine = RAGEngine()

# Global policy mode store
active_policy = {"mode": "Balanced"}


class RAGQueryRequest(BaseModel):
    query: str


class PolicyChangeRequest(BaseModel):
    mode: str


H100_SPECIFICATION = {
    "gpu_model": "NVIDIA H100 SXM5 80GB",
    "architecture": "Hopper Architecture",
    "max_tdp_w": 700.0,
    "power_limit_range_w": [300.0, 700.0],
    "throttle_temp_c": 90.0,
    "hard_shutdown_temp_c": 105.0,
    "optimal_temp_range_c": [60.0, 78.0],
    "tensor_cores": "528 Tensor Cores (4th Gen Transformer Engine FP8/FP16)",
    "memory": "80GB HBM3 @ 3.35 TB/sec",
    "parameters_monitored": [
        "GPU_Utilization_Pct",
        "GPU_Power_W",
        "GPU_Temp_C",
        "GPU_Clock_MHz",
        "Memory_Clock_MHz",
        "Memory_Util_Pct",
        "Tensor_Core_Util_Pct",
        "Fan_Speed_Pct",
        "Ambient_Temp_C",
        "Power_Limit_W",
    ],
    "control_actions": [
        "INCREASE_FAN_COOLING (Modulate Pump/Fan speed)",
        "REDUCE_POWER_LIMIT (Cap TDP from 700W down to 300W)",
        "MAINTAIN_SETTINGS (Sustain efficient baseline)",
    ]
}


@router.get("/h100/status")
async def get_h100_status():
    """Returns NVIDIA H100 specifications, operating limits, and current AI Policy mode."""
    res = H100_SPECIFICATION.copy()
    res["active_policy_mode"] = active_policy["mode"]
    return res


@router.get("/rag/knowledge")
async def get_rag_knowledge_documents():
    """Returns the list of documents indexed in the H100 RAG Vector Knowledge Base."""
    return {"documents_count": len(H100_KNOWLEDGE_DOCUMENTS), "documents": H100_KNOWLEDGE_DOCUMENTS}


@router.post("/rag/query")
async def query_rag_engine(req: RAGQueryRequest):
    """Executes vector retrieval query over H100 Knowledge Base and returns cited response."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    result = rag_engine.query(req.query)
    return result


@router.get("/agent/policy")
async def get_agent_policy():
    """Returns the active Agentic AI Control Policy mode."""
    return {"mode": active_policy["mode"]}


@router.post("/agent/policy")
async def set_agent_policy(req: PolicyChangeRequest):
    """Updates the active Agentic AI Control Policy mode."""
    valid_modes = ["Balanced", "Max_Performance", "Eco_Silent", "Strict_Thermal_Cap"]
    if req.mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid policy mode. Must be one of {valid_modes}")
    active_policy["mode"] = req.mode
    return {"status": "success", "mode": active_policy["mode"]}