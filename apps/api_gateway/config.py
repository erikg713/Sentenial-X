import os
from pathlib import Path

# Server settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Database configuration (PostgreSQL example)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "sentenial_db")
DB_USER = os.getenv("DB_USER", "sentenial_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "supersecret")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Authentication settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change_this_secret_key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# External services (optional)
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "https://api.example.com")

# Feature toggles
ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"

# Other configs
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", 10485760))  # 10 MB default
