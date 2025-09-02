import torch
from .model import ThreatClassifier
from .config import settings
from .preprocess import preprocess_input
from .utils import load_model
import logging

logger = logging.getLogger("sentenialx.MLInfer")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(settings.LOG_LEVEL)

class MLInfer:
    """
    ML inference class for threat detection.
    """

    def __init__(self, model_path=None):
        self.device = settings.DEVICE
        self.model_path = model_path or settings.MODEL_PATH
        self.model = ThreatClassifier().to(self.device)
        self.model = load_model(self.model, self.model_path)
        self.model.eval()

    def predict(self, data: dict) -> dict:
        """
        Predict threat type and confidence.
        """
        features = preprocess_input(data)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(features_tensor)
            confidence, predicted = torch.max(output, 1)
        result = {
            "predicted_class": int(predicted.item()),
            "confidence": float(confidence.item())
        }
        logger.info(f"Prediction result: {result}")
        return result
