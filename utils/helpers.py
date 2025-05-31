# utils/helpers.py
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='sentenial_x_utils.log', filemode='a')

def load_config(config_path='config.json'):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}

def get_timestamp():
    """Get current timestamp."""
    return time.strftime('%Y-%m-%d %H:%M:%S')
