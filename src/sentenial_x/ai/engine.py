# src/sentenial_x/ai/engine.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMEngine:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                      .to(self.device))
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
