import logging

logger = logging.getLogger("custom_scanners")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def log_scan(scanner_name: str, target: str, result: dict):
    logger.info("Scanner %s ran on target %s with result: %s", scanner_name, target, result)
