import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    FLASK_APP = os.getenv("FLASK_APP", "sentinel_core.py")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_PATH = os.getenv("LOG_PATH", "analytics/memory_scan_logs/emulation.log")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # API / External services
    THREAT_INTEL_API_KEY = os.getenv("THREAT_INTEL_API_KEY")
    EXFILTRATION_ENDPOINT = os.getenv("EXFILTRATION_ENDPOINT")

    # Emulation
    POWERSHELL_EXECUTABLE = os.getenv("POWERSHELL_EXECUTABLE", "pwsh")

    # Mail
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET = os.getenv("JWT_SECRET")

    # Feature Flags
    ENABLE_DEEP_EMULATION = os.getenv("ENABLE_DEEP_EMULATION", "false").lower() == "true"
    ENABLE_GUI_DASHBOARD = os.getenv("ENABLE_GUI_DASHBOARD", "false").lower() == "true"