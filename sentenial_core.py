import os
import logging.config
import logging
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from config.config import Config

# Import our custom modules
from ai_core.adversarial_detector import AdversarialDetector
from envs.cyber_env import create_environment

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants (Read from environment) ---
LOG_CONFIG_PATH = os.getenv('LOG_CONFIG_PATH', 'etc/logging.conf')
ENABLE_DEEP_EMULATION = os.getenv("ENABLE_DEEP_EMULATION", 'false').lower() == 'true'

# --- Application Factory ---
def create_app():
    # Use instance_relative_config=True if you need instance-specific files
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "supersecret"),
        JWT_SECRET=os.getenv("JWT_SECRET", "mydevjwtkey"),
        DATABASE_URL=os.getenv("DATABASE_URL"),
        THREAT_INTEL_API_KEY=os.getenv("THREAT_INTEL_API_KEY")
    )

    # 1. Setup Logging
    if os.path.exists(LOG_CONFIG_PATH):
        try:
            # Load configuration from INI file
            logging.config.fileConfig(LOG_CONFIG_PATH, disable_existing_loggers=False)
        except Exception as e:
            # Fallback for configuration errors
            logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
            logging.error(f"Failed to load logging configuration from file: {e}")

    # Set the Flask logger to our configured logger named 'sentenial_core'
    app.logger = logging.getLogger("sentenial_core")
    app.logger.info("Sentenial Core application starting...")

    # 2. Gunicorn Integration (Production Logging Best Practice)
    if __name__ != '__main__':
        # If running under Gunicorn, use Gunicorn's handlers and level
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
        app.logger.info("Integrated with Gunicorn logging.")

    # 3. Initialize Subsystems
    app.detector = AdversarialDetector(config={'ADVERSARIAL_THRESHOLD': 0.8})

    if ENABLE_DEEP_EMULATION:
        try:
            app.cyber_env = create_environment()
            app.logger.info("Deep Emulation environment ready.")
        except Exception as e:
            app.logger.error(f"Failed to initialize Cyber Environment: {e}")
            app.cyber_env = None

    # 4. Define Routes
    @app.route('/health', methods=['GET'])
    def health_check():
        """Basic health check endpoint."""
        return jsonify({"status": "OK", "name": "Sentenial Core", "env": app.config['SECRET_KEY'][:6] + '...'}), 200

    @app.route('/scan', methods=['POST'])
    def scan_file():
        """Endpoint to receive data for analysis."""
        data = request.get_json(silent=True)
        if not data:
            app.logger.warning("Scan attempted with no JSON payload.")
            return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

        # Run adversarial detection
        input_data_mock = str(data) # In a real app, this would be the raw file/input data tensor
        detection_result = app.detector.detect(input_data_mock)
        
        if detection_result['is_adversarial']:
            app.logger.error("Malicious input detected by Adversarial Detector.")
            return jsonify({"status": "blocked", "reason": "Adversarial input detected", "confidence": detection_result['confidence']}), 403

        # Add primary security analysis here...
        app.logger.info("Input passed adversarial check. Beginning core analysis.")
        return jsonify({"status": "analyzed", "threat_level": 1, "details": "Core analysis complete."})

    return app

# Gunicorn/WSGI entry point
app = create_app()

if __name__ == '__main__':
    # Flask development server runs when called via `python sentenial_core.py`
    app.run(debug=True, host='0.0.0.0', port=5000)
