# sentenialx/models/lora/qlora_tuner.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer  # For supervised fine-tuning
from sentenialx.models.artifacts import register_artifact
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def tune_qlora(base_model_name: str = "meta-llama/Llama-2-7b-hf",  # Requires HF access
               dataset_path: str = "sentenialx/data/processed/threat_intents.jsonl",
               rank: int = 64, alpha: int = 16, dropout: float = 0.05,
               bits: int = 4, output_dir: str = "sentenialx/models/artifacts/lora/qlora_weights_v1"):
    """
    QLoRA fine-tuning for large models.
    - base_model_name: HF model ID (gated models need token).
    - dataset_path: JSONL with 'text' (prompt) and 'label' (or formatted for SFT).
    - rank: LoRA rank (higher for larger models).
    """
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
        bnb_4bit_use_double_quant=True,  # Double quant
        bnb_4bit_quant_storage=torch.uint8
    )
    
    # Load tokenizer and quantized model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Paged attention
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)  # Gradient checkpointing
    
    # LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],  # Llama-specific
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # ~0.1% params
    
    # Dataset (SFT format: e.g., {"text": "<prompt>### Response: <label>"})
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Fits in memory
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",  # Paged optimizer
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,  # Or fp16
        report_to="none"
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512
    )
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    artifact_path = Path(output_dir) / "adapter_model.bin"
    metadata = {
        "base_model": base_model_name,
        "quant_bits": bits,
        "rank": rank,
        "dataset": dataset_path,
        "eval_loss": trainer.evaluate()["eval_loss"]
    }
    register_artifact("lora", artifact_path, "1.0.0", metadata)  # Reuse 'lora' type for QLoRA
    logging.info("QLoRA tuning complete and registered.")

if __name__ == "__main__":
    tune_qlora()
