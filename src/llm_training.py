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
