# config/config.py

"""
Sentenial-X Configuration Module
--------------------------------
Holds environment-based settings for API, CLI, AI Core, and Dashboard.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file at project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Centralized configuration for Sentenial-X platform."""

    # API
    API_TITLE: str = "Sentenial-X API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready API for Sentenial-X cybersecurity platform"
    API_KEY: str = os.getenv("API_KEY", "super-secret-key")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 4))

    # Database
    DB_URI: str = os.getenv("DB_URI", "postgresql://user:pass@localhost:5432/sentenialx")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # AI Core
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "ai_core/models"))
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 768))
    THREAT_THRESHOLD: int = int(os.getenv("THREAT_THRESHOLD", 80))

    # Dashboard
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", 3000))


# Singleton instance
settings = Settings()
