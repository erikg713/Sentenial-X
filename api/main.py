# api/main.py
"""
Sentenial-X API Main Entrypoint
Production-ready FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import telemetry, orchestrator, cortex, wormgpt, exploits
from api.utils.logger import init_logger

# Initialize logger
logger = init_logger("sentenialx_api")

# Initialize FastAPI app
app = FastAPI(
    title="Sentenial-X API",
    description="Production-ready API layer for the Sentenial-X cybersecurity platform",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(orchestrator.router, prefix="/api/orchestrator", tags=["Orchestrator"])
app.include_router(cortex.router, prefix="/api/cortex", tags=["Cortex"])
app.include_router(wormgpt.router, prefix="/api/wormgpt", tags=["WormGPT"])
app.include_router(exploits.router, prefix="/api/exploits", tags=["Exploits"])

# Root endpoint for health checks
@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint
    """
    logger.info("Health check requested")
    return {"status": "Sentenial-X API is running 🚀"}

# Optional startup/shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Sentenial-X API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Sentenial-X API shutting down...")
