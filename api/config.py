import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "changeme")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    DB_URL = os.getenv("DB_URL", "sqlite:///sentenialx.db") 