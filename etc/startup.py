# etc/startup.py

import os
import sys
import logging
from dotenv import load_dotenv

# Use the core logger for startup messages
logger = logging.getLogger("sentenial_core")

def validate_environment():
    """
    Performs critical checks on required environment variables.
    Exits the process if validation fails to prevent launching with insecure or incomplete config.
    """
    load_dotenv(override=True)
    
    # 1. Check Flask/Security Critical Variables
    required_vars = ["SECRET_KEY", "JWT_SECRET", "DATABASE_URL"]
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            logger.error(f"FATAL CONFIGURATION ERROR: Required environment variable '{var}' is missing.")
            sys.exit(1)
        
        # Check for weak development keys in production environments
        if os.getenv("FLASK_ENV") == "production":
            if "supersecret" in value.lower() or "devkey" in value.lower():
                logger.error(f"FATAL SECURITY ERROR: Variable '{var}' is set to a weak development value.")
                sys.exit(1)

    # 2. Check Feature-Specific Variables
    if os.getenv("ENABLE_DEEP_EMULATION", 'false').lower() == 'true':
        # Check if the environment path for logs is writable
        log_path = os.getenv("LOG_PATH")
        if not log_path:
            logger.warning("LOG_PATH is not set, Deep Emulation logs will default to stdout.")
        else:
            # Attempt to create the parent directory for the log file
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    logger.info(f"Created log directory: {log_dir}")
                except OSError as e:
                    logger.error(f"FATAL: Cannot create log directory '{log_dir}'. Check permissions. Error: {e}")
                    sys.exit(1)

    logger.info("Environment validation successful. Starting Sentenial-X service.")

if __name__ == "__main__":
    # Note: In a real microservice, this is often called as a pre-command 
    # before the main Gunicorn/uvicorn command.
    validate_environment()
