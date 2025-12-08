# etc/startup.py

import os
import sys
import logging
import yaml
from dotenv import load_dotenv

# Use the core logger for startup messages
logger = logging.getLogger("sentenial_core")

def load_and_validate():
    """
    Loads ENV variables and configuration, validating critical security settings.
    """
    # 1. Load ENV Variables
    load_dotenv(override=True)
    
    # 2. Load YAML Configuration
    config_path = os.getenv('LOG_CONFIG_PATH', 'etc/config.yaml')
    try:
        with open(config_path, 'r') as f:
            app_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"FATAL: Configuration file not found at {config_path}.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"FATAL: Error parsing YAML config file: {e}")
        sys.exit(1)

    # 3. Critical Security Check
    required_secrets = ["SECRET_KEY", "JWT_SECRET"]
    
    for var in required_secrets:
        value = os.getenv(var)
        if not value or "replace" in value.lower():
            logger.error(f"FATAL SECURITY ERROR: Variable '{var}' is missing or set to a placeholder.")
            sys.exit(1)
    
    # 4. Log Directory Check
    log_path = os.getenv("LOG_PATH")
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created log directory: {log_dir}")
            except OSError as e:
                logger.error(f"FATAL: Cannot create log directory '{log_dir}'. Check permissions. Error: {e}")
                sys.exit(1)

    logger.info("Environment and configuration validation successful.")
    return app_config

if __name__ == "__main__":
    # Ensure logging is minimally configured before running validation
    logging.basicConfig(level=logging.INFO)
    load_and_validate()
