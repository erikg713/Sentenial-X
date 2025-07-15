import os
from dotenv import load_dotenv

# Load environment-specific file
env_file = f".env.{os.getenv('FLASK_ENV', 'development')}"
load_dotenv(dotenv_path=env_file)

class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    DATABASE_URL = os.getenv("DATABASE_URL")
    JWT_SECRET = os.getenv("JWT_SECRET")

    # Optional features
    ENABLE_GUI = os.getenv("ENABLE_GUI_DASHBOARD", "false").lower() == "true"
    ENABLE_DEEP_EMULATION = os.getenv("ENABLE_DEEP_EMULATION", "false").lower() == "true"
