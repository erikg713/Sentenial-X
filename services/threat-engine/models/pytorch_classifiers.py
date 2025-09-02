import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleThreatClassifier(nn.Module):
    """
    A simple feedforward neural network for threat classification.
    - Input: numerical features or embeddings from NLP/vectorizer
    - Output: probabilities across threat categories
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super(SimpleThreatClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNThreatClassifier(nn.Module):
    """
    A CNN-based text threat classifier (good for log/event payloads).
    - Embeddings -> Conv layers -> Dense classification
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int = 2):
        super(CNNThreatClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * ((100 - 5 + 1) // 2), 64)  # assuming max_len=100
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_model(model_type: str, **kwargs):
    """
    Factory function to load the requested model.
    """
    if model_type == "simple":
        return SimpleThreatClassifier(**kwargs)
    elif model_type == "cnn":
        return CNNThreatClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
