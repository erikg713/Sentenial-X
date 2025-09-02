import torch
import torch.nn as nn
from .config import settings

class ThreatClassifier(nn.Module):
    """
    Simple fully connected neural network for threat classification.
    """

    def __init__(self):
        super(ThreatClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(settings.INPUT_FEATURES, settings.HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(settings.HIDDEN_UNITS, settings.HIDDEN_UNITS // 2),
            nn.ReLU(),
            nn.Linear(settings.HIDDEN_UNITS // 2, settings.OUTPUT_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)
