import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Settings:
    """Configuration for ML Threat Detector"""

    # Model
    MODEL_NAME = "threat_classifier.pt"
    MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
    INPUT_FEATURES = 128  # Example feature size
    HIDDEN_UNITS = 256
    OUTPUT_CLASSES = 5  # e.g., malware, phishing, rce, xss, normal

    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Thresholds
    DETECTION_THRESHOLD = 0.7

    # Logging
    LOG_LEVEL = "INFO"

    # Dataset
    DATASET_PATH = BASE_DIR / "dataset"

    # Device
    DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

settings = Settings()
