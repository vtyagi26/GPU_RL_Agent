from fastapi import APIRouter

router = APIRouter()

GPU_TIERS = {
    "H100": {
        "id": "H100",
        "name": "NVIDIA H100 SXM (Enterprise Datacenter)",
        "type": "Liquid Cooled",
        "parameters_count": 8,
        "max_power_w": 700,
        "throttle_temp_c": 90,
        "parameters": [
            "GPU_Util (%)", "VRAM_Util (%)", "Core_Clock (MHz)", "Power_Draw (W)",
            "Coolant_Inlet_Temp (°C)", "Coolant_Flow_Rate (L/min)", "Die_Temp (°C)", "Ambient_Room_Temp (°C)"
        ],
        "controls": ["Coolant Flow Rate (LPM)", "Fan Speed (%)", "Power Limit (W)"]
    },
    "RTX6000": {
        "id": "RTX6000",
        "name": "NVIDIA RTX 6000 Ada (Workstation)",
        "type": "Multi-Fan Air Cooled",
        "parameters_count": 5,
        "max_power_w": 300,
        "throttle_temp_c": 85,
        "parameters": [
            "GPU_Util (%)", "Core_Clock (MHz)", "Power_Draw (W)", "Fan_Speed (RPM)", "Die_Temp (°C)"
        ],
        "controls": ["Fan Speed (RPM)", "Power Limit (W)"]
    },
    "RTX4050": {
        "id": "RTX4050",
        "name": "NVIDIA RTX 4050 (Consumer Laptop)",
        "type": "Shared Heatsink Laptop Air",
        "parameters_count": 4,
        "max_power_w": 85,
        "throttle_temp_c": 87,
        "parameters": [
            "GPU_Util (%)", "Power_Draw (W)", "Shared_Fan_Speed (RPM)", "Die_Temp (°C)"
        ],
        "controls": ["Shared Fan Speed (RPM)"]
    }
}

@router.get("/gpus")
async def get_available_gpus():
    """Returns the list of available GPU profiles and their control specifications."""
    return {"gpus": list(GPU_TIERS.values())}

@router.get("/gpus/{tier_id}")
async def get_gpu_details(tier_id: str):
    """Returns specific profile info for a single GPU tier."""
    tier_info = GPU_TIERS.get(tier_id.upper())
    if not tier_info:
        return {"error": "GPU Tier not found"}, 404
    return tier_info