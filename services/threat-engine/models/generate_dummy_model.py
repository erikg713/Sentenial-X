import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SimpleThreatClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=2):
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

def build_and_save_model(output_path="threat_model.pkl"):
    model = SimpleThreatClassifier()
    for param in model.parameters():
        if param.requires_grad:
            nn.init.normal_(param, mean=0, std=0.02)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Dummy threat model saved at: {output_path}")

if __name__ == "__main__":
    model_dir = "services/threat-engine/models"
    os.makedirs(model_dir, exist_ok=True)
    build_and_save_model(os.path.join(model_dir, "threat_model.pkl"))
