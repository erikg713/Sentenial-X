# sentenialx/models/lora/lora_tuner.py
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset
from sentenialx.models.artifacts import register_artifact
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def tune_lora(base_model_name: str = "bert-base-uncased", 
              dataset_path: str = "sentenialx/data/processed/threat_intents.csv",
              rank: int = 8, alpha: int = 16, dropout: float = 0.1,
              output_dir: str = "sentenialx/models/artifacts/lora/lora_weights_v1"):
    """
    Fine-tune with LoRA.
    - base_model_name: Pre-trained model (from HF or artifact registry).
    - dataset_path: CSV with 'text' and 'label' columns.
    - rank: LoRA rank (low for efficiency).
    """
    # Load dataset
    dataset = load_dataset("csv", data_files=dataset_path)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    
    # Load base model (int8 for efficiency)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5)
    model = prepare_model_for_int8_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=rank,  # Rank
        lora_alpha=alpha,  # Scaling
        target_modules=["query", "value"],  # Attention layers
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS"  # Classification for threats
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Logs % params trained (~0.1%)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none"  # No external logging for security
    )
    
    # Tokenizer (placeholder)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    dataset = dataset.map(tokenize, batched=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    artifact_path = Path(output_dir) / "adapter_model.bin"
    metadata = {
        "base_model": base_model_name,
        "rank": rank,
        "dataset": dataset_path,
        "eval_accuracy": trainer.evaluate()["eval_accuracy"]
    }
    register_artifact("lora", artifact_path, "1.0.0", metadata)
    logging.info("LoRA tuning complete and registered.")

if __name__ == "__main__":
    tune_lora()
