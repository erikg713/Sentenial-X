from fastapi import FastAPI
from api.routes import threats, exploits, monitor, orchestrator

app = FastAPI(
    title="Sentenial-X API",
    description="REST API for threat emulation, monitoring, and orchestration",
    version="1.0.0"
)

# Register routes
app.include_router(threats.router, prefix="/api/threats", tags=["Threats"])
app.include_router(exploits.router, prefix="/api/exploits", tags=["Exploits"])
app.include_router(monitor.router, prefix="/api/monitor", tags=["Monitoring"])
app.include_router(orchestrator.router, prefix="/api/orchestrator", tags=["Orchestrator"])

@app.get("/")
async def root():
    return {"status": "Sentenial-X API is running"} 