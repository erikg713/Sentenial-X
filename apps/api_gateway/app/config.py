# app/config.py
from pydantic import BaseSettings, AnyHttpUrl


class Settings(BaseSettings):
    APP_NAME: str = "Sentenial X API Gateway"
    ENV: str = "development"
    DEBUG: bool = True

    # JWT settings
    JWT_SECRET: str = "change-me-in-prod"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_SECONDS: int = 3600

    # CORS
    FRONTEND_ORIGINS: list[AnyHttpUrl] = ["http://localhost:3000"]

    # Upstream services (example)
    THREAT_ENGINE_URL: AnyHttpUrl | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()