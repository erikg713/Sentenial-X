# sentenial_core.py

import os
import logging.config
import logging
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import yaml

# Import our custom modules
# NOTE: The actual import path might need adjustment based on where 'ai_core' is relative to 'sentenial_core.py'
from ai_core.adversarial_detector import AdversarialDetector
from envs.cyber_env import create_environment # Assumes envs/ is accessible

# Load environment variables
load_dotenv()

# --- Load Configuration ---
CONFIG_PATH = os.getenv('LOG_CONFIG_PATH', 'etc/config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        APP_CONFIG = yaml.safe_load(f)
except Exception:
    APP_CONFIG = {} # Fallback

# --- Application Factory ---
def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY"),
        JWT_SECRET=os.getenv("JWT_SECRET"),
        DATABASE_URL=os.getenv("DATABASE_URL"),
        THREAT_INTEL_API_KEY=os.getenv("THREAT_INTEL_API_KEY"),
        CONFIG=APP_CONFIG
    )

    # 1. Setup Logging
    log_config_path = os.getenv('LOG_CONFIG_PATH', 'etc/logging.conf')
    if os.path.exists(log_config_path):
        try:
            logging.config.fileConfig(log_config_path, disable_existing_loggers=False)
        except Exception:
            logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    
    app.logger = logging.getLogger("sentenial_core")
    app.logger.info("Sentenial Core application starting...")

    # 2. Gunicorn Integration (Production Logging)
    if __name__ != '__main__':
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)

    # 3. Initialize Subsystems
    threshold = APP_CONFIG.get('threat_engine', {}).get('scoring_threshold', 0.85)
    app.detector = AdversarialDetector(config={'ADVERSARIAL_THRESHOLD': threshold})

    if os.getenv("ENABLE_DEEP_EMULATION", 'false').lower() == 'true':
        try:
            app.cyber_env = create_environment()
            app.logger.info("Deep Emulation environment ready.")
        except Exception as e:
            app.logger.error(f"Failed to initialize Cyber Environment: {e}")
            app.cyber_env = None

    # 4. Define Routes
    @app.route('/scan', methods=['POST'])
    def scan_file():
        data = request.get_json(silent=True)
        if not data: return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

        # Run adversarial detection
        input_data_mock = str(data) 
        detection_result = app.detector.detect(input_data_mock)
        
        if detection_result['is_adversarial']:
            app.logger.error(f"Input blocked. Adversarial score: {detection_result['confidence']}.")
            return jsonify({"status": "blocked", "reason": "Adversarial input detected"}), 403

        app.logger.info("Input passed adversarial check. Beginning core analysis.")
        return jsonify({"status": "analyzed", "threat_level": 1})

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
