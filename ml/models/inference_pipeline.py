# sentenialx/models/inference_pipeline.py

from pathlib import Path
from sentenialx.models.encoder.text_encoder import ThreatTextEncoder
from sentenialx.models.lora.lora_loader import load_lora_model
from sentenialx.models.distill.student_model import DistilledThreatModel


def build_inference_pipeline(mode: str = "distilled"):
    """
    Factory to build a text → embedding → model inference pipeline.

    mode:
      "full"      → teacher model with LoRA weights
      "distilled" → lightweight distilled student model
    """
    encoder = ThreatTextEncoder()

    if mode == "full":
        model_path = Path("sentenialx/models/base/threat_model.pt")
        lora_path = Path("sentenialx/models/lora/lora_weights.bin")
        if not model_path.exists() or not lora_path.exists():
            raise FileNotFoundError("Full LoRA mode selected, but weights are missing.")
        model = load_lora_model(str(model_path), str(lora_path))

    elif mode == "distilled":
        student_path = Path("sentenialx/models/distill/student_model.onnx")
        if not student_path.exists():
            raise FileNotFoundError("Distilled mode selected, but student model is missing.")
        model = DistilledThreatModel.load(str(student_path))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    def pipeline(input_text):
        # Ensure consistent list input to the encoder
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = list(input_text)
        features = encoder.encode(input_texts)
        output = model(features)
        return output

    return pipeline
