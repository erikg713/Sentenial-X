import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenialXNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=2, dropout=0.3):
        super(SentenialXNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
