import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from http_encoder import HTTPEncoder
from utils import load_config

class ModelManager:
    def __init__(self, config_path: str = "config.yml"):
        cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load base LLM (quantized)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, use_fast=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.model.base,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Attach LoRA adapters
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj"]
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm.load_adapter(cfg.model.lora_path)

        # HTTP traffic encoder
        self.http_encoder = HTTPEncoder(
            pretrained=cfg.http_encoder.pretrained,
            embed_dim=cfg.http_encoder.embed_dim
        ).to(self.device)

        self.max_new_tokens = cfg.server.max_new_tokens

    def generate(
        self,
        prompt: str,
        http_json: str = None
    ) -> str:
        """
        1. Optionally encode HTTP JSON into embedding
        2. Fuse by prepending a special token + embed (cross-attn)
           Here we simplify by appending JSON text to prompt.
        3. Generate a completion.
        """
        if http_json:
            # In a full implementation, fuse via cross-attention adapters.
            prompt = f"{prompt}\n\n<HTTP_PAYLOAD> {http_json}"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to(self.device)

        out = self.llm.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

