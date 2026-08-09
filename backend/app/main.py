from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.api.websockets import ws_router

app = FastAPI(
    title="NVIDIA H100 Agentic AI Thermal Controller & RAG Engine",
    description="Backend API powering real-time 10-parameter H100 thermal forecasting, RAG diagnosis, and 5-stage Agentic Control Policy.",
    version="2.0.0"
)

# Allow CORS for React frontend (Vite defaults to http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "NVIDIA H100 Agentic AI Thermal Controller",
        "message": "Backend API & RAG Vector Engine is running."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)