from models.encoder.text_encoder import ThreatTextEncoder
from models.lora.lora_loader import load_lora_model
from models.distill.student_model import DistilledThreatModel

def build_inference_pipeline(mode="distilled"):
    # Step 1: Encode input features
    encoder = ThreatTextEncoder()

    if mode == "full":
        # Load full LoRA-adapted teacher model
        model = load_lora_model("models/base/threat_model.pt", "models/lora/weights.bin")
    elif mode == "distilled":
        # Load pre-trained student model
        model = DistilledThreatModel.load("models/distill/threat_student.pt")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def pipeline(input_text):
        features = encoder.encode(input_text)
        output = model(features)
        return output

    return pipeline 
