# sentenial_core/interfaces/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    # Kafka settings
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "pinet-threat-logs")
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "sentenial-group")

    # WebSocket settings
    WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8080/logs")

    # PiNet logs
    PINET_LOG_PATH = os.getenv("PINET_LOG_PATH", "/var/log/pinet/pinet.log")

    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "./saved_models/cyber_bert")
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu")
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))

    # API server
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Other generic configs can go here

config = Config()

