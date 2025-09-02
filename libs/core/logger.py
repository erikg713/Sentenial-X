import logging
import os
import json
from logging.handlers import RotatingFileHandler

try:
    from colorlog import ColoredFormatter
    COLORLOG_AVAILABLE = True
except ImportError:
    print("[LOGGING] Optional: Install 'colorlog' for colored logs.")
    COLORLOG_AVAILABLE = False

# Defaults
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "sentenialx.log"
MAX_BYTES = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3


def _load_module_levels(config_path: str) -> dict:
    """
    Load per-module log levels from JSON file.
    Returns an empty dict if file not found or on parse errors.
    """
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[LOGGING] Failed to load log config '{config_path}': {e}")
        return {}


def _get_plain_formatter() -> logging.Formatter:
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] :: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )


def _get_colored_formatter() -> ColoredFormatter:
    return ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s] :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )


class LoggerFactory:
    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        log_file: str = DEFAULT_LOG_FILE,
        max_bytes: int = MAX_BYTES,
        backup_count: int = BACKUP_COUNT,
        config_path: str = os.getenv("LOG_CONFIG_FILE", "config/logging_levels.json")
    ):
        self.log_dir = log_dir
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.module_levels = _load_module_levels(config_path)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_logger(self, name: str, default_level: str = "INFO") -> logging.Logger:
        """
        Returns a configured logger. Removes only handlers it created previously.
        """
        logger = logging.getLogger(name)

        # Determine level: override per module-prefix if set
        module_base = name.split(".")[0]
        level_name = self.module_levels.get(module_base, default_level).upper()

        # Validate level name and get numeric value
        numeric_level = logging.getLevelName(level_name)
        if not isinstance(numeric_level, int):
            print(f"[LOGGING] Invalid log level '{level_name}' for '{name}', defaulting to INFO.")
            numeric_level = logging.INFO

        logger.setLevel(numeric_level)

        # Remove existing StreamHandler or RotatingFileHandler to avoid duplicates
        for h in list(logger.handlers):
            if isinstance(h, (logging.StreamHandler, RotatingFileHandler)):
                logger.removeHandler(h)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(_get_colored_formatter() if COLORLOG_AVAILABLE else _get_plain_formatter())
        logger.addHandler(ch)

        # File handler (rotating)
        fh = RotatingFileHandler(
            os.path.join(self.log_dir, self.log_file),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        fh.setFormatter(_get_plain_formatter())
        logger.addHandler(fh)

        return logger


# Singleton factory for module-level usage
_default_factory = LoggerFactory()


def setup_logger(name: str = "sentenialx", default_level: str = "INFO") -> logging.Logger:
    return _default_factory.get_logger(name, default_level)


if __name__ == "__main__":
    # Example usage
    os.environ["LOG_CONFIG_FILE"] = "config/logging_levels.json"
    setup_logger("core.analyzer").debug("Core module debug")
    setup_logger("api.gateway").info("API module info")
    setup_logger("engine.detector").warning("Engine warning")
    setup_logger("telemetry.receiver").error("Telemetry error")
