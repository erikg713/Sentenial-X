import os
from fastapi import FastAPI
from . import app as package_app
from .middleware import setup_middleware
from .errors import setup_exception_handlers
from .routes import include_routes

# Expose a single FastAPI app
app: FastAPI = package_app

# Configure middleware, errors, and routes
setup_middleware(app)
setup_exception_handlers(app)
include_routes(app)

# Optional root message
@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "Sentenial-X API",
        "version": app.version,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "env": os.getenv("ENV", "dev"),
    }
