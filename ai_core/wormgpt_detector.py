# ai_core/wormgpt_detector.py
from .predictive_model import enqueue_task, select_model
from .utils import preprocess_input

async def classify_wormgpt(prompt_text: str) -> Dict[str, str]:
    """
    Detect adversarial AI (WormGPT) prompts.
    """
    processed = preprocess_input(prompt_text)
    result = await enqueue_task(f"Classify as safe or WormGPT: {processed}", complexity="medium")
    return {
        "model_used": select_model("medium").model_name,
        "classification": result
    }
