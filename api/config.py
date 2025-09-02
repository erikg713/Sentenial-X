# api/config.py
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    """Central configuration for Sentenial-X API."""

    # API Metadata
    API_TITLE: str = "Sentenial-X API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready API for Sentenial-X cybersecurity platform"
    API_KEY: str = os.getenv("API_KEY", "super-secret-key")  # override in prod via env

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 4))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

    # Database
    DB_URL: str = os.getenv("DATABASE_URL", "sqlite:///sentenialx.db")  # or Postgres URL

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Internal
    START_TIME: float = time.time()  # used for uptime calculations

# Singleton instance
settings = Settings()
