from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import telemetry, orchestrator, cortex, wormgpt, exploits
from api.utils.logger import init_logger

logger = init_logger("sentenialx_api")

app = FastAPI(
    title="Sentenial-X API",
    description="Production-ready API layer for the Sentenial-X cybersecurity platform",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(orchestrator.router, prefix="/api/orchestrator", tags=["Orchestrator"])
app.include_router(cortex.router, prefix="/api/cortex", tags=["Cortex"])
app.include_router(wormgpt.router, prefix="/api/wormgpt", tags=["WormGPT"])
app.include_router(exploits.router, prefix="/api/exploits", tags=["Exploits"])

@app.get("/")
async def root():
    return {"status": "Sentenial-X API is running 🚀"}
