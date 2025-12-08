# CORE APPLICATION SETTINGS (Used by sentenial_core, threat-engine, etc.)
FLASK_ENV=development
FLASK_DEBUG=1
# The core entry file is likely handled by Docker Compose now, but included for context
# FLASK_APP=sentenial_core.py 

# CRITICAL SECURITY KEYS (MUST be changed from defaults for production!)
SECRET_KEY=A_long_random_string_for_session_signing
JWT_SECRET=Another_long_random_string_for_JWTs

# LOGGING
LOG_LEVEL=DEBUG
LOG_PATH=/app/analytics/memory_scan_logs/emulation.log
LOG_CONFIG_PATH=etc/logging.conf
