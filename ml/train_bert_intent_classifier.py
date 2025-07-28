from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Example custom dataset
data = {
    "text": [
        "Root access gained via buffer overflow.",
        "User downloaded unknown executable.",
        "Suspicious outbound connection to port 4444.",
        "System booted normally.",
        "User logged in via SSH from known device."
    ],
    "label": [0, 1, 2, 3, 3]  # 0=breach, 1=malware, 2=exploit, 3=benign
}

label_names = ["breach", "malware", "exploit", "benign"]

dataset = Dataset.from_dict(data)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_names))

args = TrainingArguments(
    output_dir="./bert-threat-intent",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("bert-threat-intent")
tokenizer.save_pretrained("bert-threat-intent")
