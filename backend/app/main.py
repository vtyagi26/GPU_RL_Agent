from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.api.websockets import ws_router

app = FastAPI(
    title="GPU Thermal Analyser API",
    description="Backend API powering real-time LSTM forecasting & RL cooling optimization across 3 GPU tiers.",
    version="1.0.0"
)

# Allow CORS for React frontend (Vite defaults to http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)

@app.get("/")
async def root():
    return {"status": "online", "message": "GPU Thermal Analyser Backend is Running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)