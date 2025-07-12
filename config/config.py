import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(dotenv_path=".env.development")

class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    FLASK_APP = os.getenv("FLASK_APP", "app.py")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_PATH = os.getenv("LOG_PATH", "analytics/memory_scan_logs/emulation.log")

    DATABASE_URL = os.getenv("DATABASE_URL", "")
    REDIS_URL = os.getenv("REDIS_URL", "")

    THREAT_INTEL_API_KEY = os.getenv("THREAT_INTEL_API_KEY", "")
    EXFILTRATION_ENDPOINT = os.getenv("EXFILTRATION_ENDPOINT", "")

    POWERSHELL_EXECUTABLE = os.getenv("POWERSHELL_EXECUTABLE", "pwsh")

    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 25))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    JWT_SECRET = os.getenv("JWT_SECRET", "superjwtsecret")

    ENABLE_DEEP_EMULATION = os.getenv("ENABLE_DEEP_EMULATION", "false").lower() == "true"
    ENABLE_GUI_DASHBOARD = os.getenv("ENABLE_GUI_DASHBOARD", "false").lower() == "true"