# models/distill/distill_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from typing import List, Dict, Optional
import os

class DistillConfig:
    """
    Configuration for distillation.
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha_ce: float = 0.5,
        alpha_mse: float = 0.5,
        max_length: int = 512,
    ):
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_mse = alpha_mse
        self.max_length = max_length

class DistillDataset(Dataset):
    """
    Dataset wrapper for text + teacher logits (optional)
    """
    def __init__(self, encodings: Dict):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

class DistillTrainer:
    """
    Handles training student model to mimic teacher model outputs.
    """
    def __init__(self, teacher_model_name: str, student_model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(self.device)
        self.student = AutoModelForCausalLM.from_pretrained(student_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    def tokenize_dataset(self, texts: List[str], max_length: int = 512):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        return DistillDataset(encodings)

    def distill_loss(self, student_logits, teacher_logits, temperature, alpha_ce, alpha_mse):
        """
        Combines cross-entropy on softened logits + MSE on logits
        """
        T = temperature
        soft_targets = nn.functional.log_softmax(teacher_logits / T, dim=-1)
        student_soft = nn.functional.log_softmax(student_logits / T, dim=-1)
        loss_ce = nn.KLDivLoss(reduction="batchmean")(student_soft, soft_targets) * (T * T)
        loss_mse = nn.MSELoss()(student_logits, teacher_logits)
        return alpha_ce * loss_ce + alpha_mse * loss_mse

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        config: DistillConfig = DistillConfig(),
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5
    ):
        self.teacher.eval()
        self.student.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.no_grad():
                    teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_outputs.logits

                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                loss = self.distill_loss(student_logits, teacher_logits, config.temperature, config.alpha_ce, config.alpha_mse)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"[DistillTrainer] Epoch {epoch+1}/{epochs} complete, last batch loss: {loss.item():.4f}")

        os.makedirs(output_dir, exist_ok=True)
        self.student.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[DistillTrainer] Distilled student model saved at {output_dir}")
