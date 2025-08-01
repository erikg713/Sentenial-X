import logging
from logging.handlers import RotatingFileHandler

def configure_logger(name: str = "RetaliationBot") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        file_handler = RotatingFileHandler("retaliation_bot.log", maxBytes=5_000_000, backupCount=3)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger