import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Lightweight text/feature encoder
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=100, embedding_dim=32):
        super(SimpleEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x should be token IDs or categorical indices
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)  # average pooling
        return F.relu(self.fc(pooled))

def build_and_save_encoder(output_path="encoder.pt"):
    model = SimpleEncoder()
    for param in model.parameters():
        if param.requires_grad:
            nn.init.normal_(param, mean=0, std=0.02)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Encoder model saved at: {output_path}")

if __name__ == "__main__":
    model_dir = "services/threat-engine/models"
    os.makedirs(model_dir, exist_ok=True)
    build_and_save_encoder(os.path.join(model_dir, "encoder.pt"))
