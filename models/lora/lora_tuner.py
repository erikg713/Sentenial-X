# models/lora/lora_tuner.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
from typing import Optional, List, Dict

class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.
    """
    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        self.r = r  # rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]

class LoRADataset(Dataset):
    """
    Simple Dataset wrapper for tokenized text.
    """
    def __init__(self, encodings: Dict):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

class LoRATuner:
    """
    LoRA Tuner for Hugging Face Transformers models.
    """
    def __init__(self, model_name: str, lora_config: LoRAConfig, device: Optional[str] = None):
        self.model_name = model_name
        self.lora_config = lora_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self._inject_lora()

    def _inject_lora(self):
        """
        Inject LoRA adapters into the target modules.
        """
        for name, module in self.model.named_modules():
            if any(target in name for target in self.lora_config.target_modules):
                self._add_lora(module)
        print("[LoRATuner] LoRA adapters injected.")

    def _add_lora(self, module: nn.Module):
        """
        Add LoRA parameters to a linear module.
        """
        if isinstance(module, nn.Linear):
            # Save original weight
            module.weight_orig = module.weight.data.clone()
            # LoRA parameters
            module.lora_A = nn.Parameter(torch.zeros((self.lora_config.r, module.in_features)))
            module.lora_B = nn.Parameter(torch.zeros((module.out_features, self.lora_config.r)))
            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)
            module.forward_orig = module.forward
            # Modified forward
            def forward_lora(x, module=module):
                return module.forward_orig(x) + (module.lora_B @ module.lora_A @ x.T).T * (self.lora_config.alpha / self.lora_config.r)
            module.forward = forward_lora

    def tokenize_dataset(self, texts: List[str], max_length: int = 512):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        return LoRADataset(encodings)

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5
    ):
        """
        Train LoRA adapters using Hugging Face Trainer.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            save_total_limit=2,
            logging_steps=50,
            save_steps=200,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()
        print(f"[LoRATuner] Training complete. Saving to {output_dir}")
        self.save_lora(output_dir)

    def save_lora(self, path: str):
        """
        Save LoRA adapter parameters only.
        """
        os.makedirs(path, exist_ok=True)
        lora_state = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_state[f"{name}.A"] = module.lora_A.detach().cpu()
                lora_state[f"{name}.B"] = module.lora_B.detach().cpu()
        torch.save(lora_state, os.path.join(path, "lora_adapters.pt"))
        print(f"[LoRATuner] LoRA adapters saved at {path}/lora_adapters.pt")

    def load_lora(self, path: str):
        """
        Load LoRA adapter parameters.
        """
        lora_state = torch.load(os.path.join(path, "lora_adapters.pt"), map_location=self.device)
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                module.lora_A.data.copy_(lora_state[f"{name}.A"])
                module.lora_B.data.copy_(lora_state[f"{name}.B"])
        print(f"[LoRATuner] LoRA adapters loaded from {path}")
