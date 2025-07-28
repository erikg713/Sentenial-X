from transformers import AdapterConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load base model & tokenizer
model_name = "mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Configure LoRA
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
lora_model = get_peft_model(base_model, lora_cfg)

# Load instruction dataset
dataset = load_dataset("json", data_files="data/processed/llm_dataset.jsonl", split="train")
dataset = dataset.map(lambda x: tokenizer(x['prompt'], truncation=True, padding="max_length"), batched=True)

# Training
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="models/lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
)
trainer = Trainer(model=lora_model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model("models/lora")

# Load encoder checkpoint
from src.encoder_training import HTTPEncoder
encoder = HTTPEncoder.from_pretrained("models/encoder")

# Register cross-attention adapter
fusion_cfg = AdapterConfig(
    mh_adapter=True,
    output_adapter=True,
    reduction_factor=16,
    non_linearity="relu"
)
lora_model.register_adapter("http_fusion", config=fusion_cfg)
lora_model.add_fusion("lora", "http_fusion")

# Distillation: generate pseudo-labels with a security-fine-tuned 13B teacher
teacher = AutoModelForCausalLM.from_pretrained("security-13b")
teacher_tokenizer = tokenizer  # same tokenizer

# Generate completions
from tqdm import tqdm
pseudo = []
for example in load_dataset("json", data_files="data/processed/llm_dataset.jsonl")['train']:
    inp = teacher_tokenizer(example['prompt'], return_tensors="pt")
    out = teacher.generate(**inp, max_new_tokens=128)
    text = teacher_tokenizer.decode(out[0], skip_special_tokens=True)
    pseudo.append({"prompt": example['prompt'], "completion": text})

# Save as JSONL and retrain LoRA model with combined data
with open("data/processed/distill_dataset.jsonl","w") as f:
    for rec in pseudo:
        f.write(json.dumps(rec)+"\n")
# Rerun Trainer on combined files: llm_dataset.jsonl + distill_dataset.jsonl
