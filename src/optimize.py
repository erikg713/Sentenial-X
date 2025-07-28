import torch
from nn_pruning.inference_model_patcher import InferenceModelPatcher
from nn_pruning.model_patcher import ModelPatcher
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("models/lora")
# Prune 15% of heads
patcher = ModelPatcher(model)
patcher.prune_heads(prune_ratio=0.15)

# Quantize to 4-bit
from bitsandbytes.optim import GlobalOptimManager
manager = GlobalOptimManager.get_instance()
quantized = manager.quantize_model(model, bits=4)

quantized.save_pretrained("models/optimized")
