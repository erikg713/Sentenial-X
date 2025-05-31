# telemetry.py
import logging
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, filename='sentenial_x_telemetry.log', filemode='a')

class TelemetryListener:
    def __init__(self, callback):
        self.callback = callback
        logging.info("TelemetryListener initialized.")

    def start(self):
        """Simulate a telemetry stream."""
        logging.info("Starting telemetry stream...")
        try:
            while True:
                # Simulate telemetry event
                event = {
                    'value': random.random(),
                    'label': random.randint(0, 1),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.callback(event)
                time.sleep(5)  # Simulate events every 5 seconds
        except KeyboardInterrupt:
            logging.info("Telemetry stream stopped.")
