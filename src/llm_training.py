from transformers import AdapterConfig

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
