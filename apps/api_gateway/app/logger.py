# app/logger.py
import logging
from logging.handlers import RotatingFileHandler
from .config import settings

LOG_FILE = "/tmp/sentenialx_api_gateway.log"

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
handler.setFormatter(formatter)

logger = logging.getLogger("sentenialx.api")
logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
logger.addHandler(handler)
logger.propagate = False