# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import health, auth_routes, proxy
from .logger import logger
from .routers import roles
app.include_router(cortex_routes.router)
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.FRONTEND_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(health.router)
app.include_router(auth_routes.router)
app.include_router(proxy.router)


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Sentenial X API Gateway")


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "env": settings.ENV}
