# apps/dashboard/run.py

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.dashboard import config
from apps.dashboard.layout import router as dashboard_router
from api.utils.logger import init_logger

logger = init_logger("dashboard_app")

app = FastAPI(
    title=config.DASHBOARD_TITLE,
    description=config.DASHBOARD_DESCRIPTION,
    version=config.DASHBOARD_VERSION,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dashboard routes
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])

@app.get("/")
async def root():
    return {"status": "Dashboard running ✅"}

if __name__ == "__main__":
    logger.info(
        f"Starting Dashboard at http://{config.HOST}:{config.PORT} "
        f"with workers={config.WORKERS}"
    )
    uvicorn.run(
        "apps.dashboard.run:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        workers=config.WORKERS,
    )
