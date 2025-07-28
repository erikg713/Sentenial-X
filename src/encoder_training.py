from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from torch import nn
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

class HTTPEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(base.config.hidden_size, 256)
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask).last_hidden_state
        cls = outputs[:,0]
        return self.proj(cls)

# Dataset loading & mapping to input_ids/labels
# Train to classify malicious vs. benign sessions

# ... Trainer boilerplate ...
