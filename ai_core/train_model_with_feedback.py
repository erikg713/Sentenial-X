"""
Sentenial-X AI Core: Train Model with Feedback
-----------------------------------------------
Handles incremental training of AI models based on feedback from
threat classification, WormGPT emulations, or orchestrator events.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from api.utils.logger import init_logger
from ai_core.model_loader import ModelLoader

logger = init_logger("ai_core.train_model_with_feedback")


class FeedbackTrainer:
    """
    Trainer class for incremental AI model training using feedback.
    """

    def __init__(self, model_name: str = "cortex_ai"):
        self.model_loader = ModelLoader()
        self.model_name = model_name
        self.training_log: List[Dict[str, Any]] = []
        self.model = self.model_loader.load_model(model_name)
        logger.info("FeedbackTrainer initialized for model: %s", model_name)

    def train(self, feedback_data: List[Dict[str, Any]], epochs: int = 1) -> Dict[str, Any]:
        """
        Train the model with provided feedback data.
        """
        logger.info("Starting training for model %s with %d samples for %d epochs", 
                    self.model_name, len(feedback_data), epochs)
        try:
            # Placeholder: Replace with real training logic
            accuracy = self._mock_training(feedback_data, epochs)

            result = {
                "model_name": self.model_name,
                "samples_trained": len(feedback_data),
                "epochs": epochs,
                "accuracy": accuracy,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.training_log.append(result)
            logger.info("Training completed: %s", result)
            return result
        except Exception as e:
            logger.exception("Failed to train model: %s", e)
            return {
                "model_name": self.model_name,
                "samples_trained": len(feedback_data),
                "epochs": epochs,
                "accuracy": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _mock_training(self, feedback_data: List[Dict[str, Any]], epochs: int) -> float:
        """
        Simulated training logic (replace with real ML code).
        """
        base_accuracy = 0.7
        improvement = min(0.3, len(feedback_data) * 0.01 * epochs)
        return round(base_accuracy + improvement, 4)

    def get_training_log(self) -> List[Dict[str, Any]]:
        """
        Returns the full training log.
        """
        return list(self.training_log)

    def clear_training_log(self):
        """
        Clears the training log.
        """
        logger.warning("Clearing training log with %d entries", len(self.training_log))
        self.training_log.clear()


# ------------------------
# CLI / Test Example
# ------------------------
if __name__ == "__main__":
    trainer = FeedbackTrainer(model_name="cortex_ai")
    sample_feedback = [
        {"threat": {"type": "rce", "source": "192.168.1.10"}, "expected_severity": "critical"},
        {"threat": {"type": "scan", "source": "192.168.1.20"}, "expected_severity": "low"},
        {"threat": {"type": "xss", "source": "192.168.1.30"}, "expected_severity": "high"},
    ]

    result = trainer.train(sample_feedback, epochs=2)
    print(result)
    print("Training log:", trainer.get_training_log())
