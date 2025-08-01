import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, filename='sentenial_x_feedback.log', filemode='a')

class FeedbackCollector:
    def __init__(self):
        self.feedback_log = []
        self.log_file = 'logs/feedback_log.json'
        if not os.path.exists('logs'):
            os.makedirs('logs')

    def store_feedback(self, detection_id, label):
        """Store analyst feedback."""
        try:
            feedback = {'detection_id': detection_id, 'label': label, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
            self.feedback_log.append(feedback)
            with open(self.log_file, 'w') as f:
                json.dump(self.feedback_log, f, indent=2)
            logging.info(f"Stored feedback: {feedback}")
        except Exception as e:
            logging.error(f"Failed to store feedback: {e}")
