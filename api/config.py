# api/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    """Central configuration for Sentenial-X API."""

    # API
    API_TITLE: str = "Sentenial-X API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready API for Sentenial-X cybersecurity platform"
    API_KEY: str = os.getenv("API_KEY", "super-secret-key")  # override in prod via env

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 4))

    # Database (placeholder, extend for Postgres, SQLite, etc.)
    DB_URL: str = os.getenv("DB_URL", "sqlite:///sentenialx.db")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Singleton instance
settings = Settings()
# api/config.py
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "super-secret-key")
START_TIME = time.time()
DB_URI = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/sentenialx")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1")
