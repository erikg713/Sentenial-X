import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from .model import ThreatClassifier
from .utils import save_model, load_dataset, evaluate_model
from .config import settings
import logging

logger = logging.getLogger("sentenialx.MLTrainer")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(settings.LOG_LEVEL)

class MLTrainer:
    """
    ML Trainer for threat classification.
    """

    def __init__(self):
        self.device = settings.DEVICE
        self.model = ThreatClassifier().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)

    def train(self):
        dataset = load_dataset(settings.DATASET_PATH)
        dataloader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
        logger.info(f"Starting training for {settings.EPOCHS} epochs")

        for epoch in range(settings.EPOCHS):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{settings.EPOCHS} | Loss: {total_loss:.4f}")

        save_model(self.model, settings.MODEL_PATH)
        evaluate_model(self.model, dataloader, self.device)
