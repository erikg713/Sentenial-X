# api/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import telemetry, orchestrator, cortex, wormgpt, exploits
from api.utils.logger import init_logger
from api.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Import routes
# -------------------------------
from api.routes import (
    telemetry,
    orchestrator,
    cortex,
    wormgpt,
    exploits,
)

# Semantic Analyzer Route
from api.routes import semantic_analyzer

# -------------------------------
# Logger
# -------------------------------
from api.utils.logger import init_logger

logger = init_logger("sentenialx_api")

# -------------------------------
# FastAPI app initialization
# -------------------------------
app = FastAPI(
    title="Sentenial-X API",
    description="Production-ready API layer for the Sentenial-X cybersecurity platform",
    version="1.0.0",
)

# -------------------------------
# Enable CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be restricted in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Register routes
# -------------------------------
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(orchestrator.router, prefix="/api/orchestrator", tags=["Orchestrator"])
app.include_router(cortex.router, prefix="/api/cortex", tags=["Cortex"])
app.include_router(wormgpt.router, prefix="/api/wormgpt", tags=["WormGPT"])
app.include_router(exploits.router, prefix="/api/exploits", tags=["Exploits"])
app.include_router(semantic_analyzer.router, prefix="/api/semantic-analyzer", tags=["Semantic Analyzer"])

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/")
async def root():
    return {"status": "Sentenial-X API is running 🚀"}

# -------------------------------
# Startup / Shutdown events
# -------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Sentenial-X API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Sentenial-X API shutdown complete")
# Initialize logger
logger = init_logger("sentenialx_api")

# Create FastAPI app with config metadata
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if settings.ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(orchestrator.router, prefix="/api/orchestrator", tags=["Orchestrator"])
app.include_router(cortex.router, prefix="/api/cortex", tags=["Cortex"])
app.include_router(wormgpt.router, prefix="/api/wormgpt", tags=["WormGPT"])
app.include_router(exploits.router, prefix="/api/exploits", tags=["Exploits"])

@app.get("/")
async def root():
    return {"status": "Sentenial-X API is running 🚀"}


# Production entrypoint
def start():
    """
    Run the API with Uvicorn.
    Reads HOST, PORT, WORKERS from settings.
    """
    logger.info(f"Starting Sentenial-X API on {settings.HOST}:{settings.PORT} with {settings.WORKERS} workers")
    uvicorn.run(
        "api.server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.WORKERS,
        reload=False  # set True for dev
    )


if __name__ == "__main__":
    start()
