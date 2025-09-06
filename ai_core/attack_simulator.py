# ai_core/attack_simulator.py
from .predictive_model import enqueue_task, select_model
from .utils import preprocess_input

async def simulate_attack(prompt_text: str) -> Dict[str, str]:
    """
    Multi-step attack simulation using 405B.
    """
    processed = preprocess_input(prompt_text)
    result = await enqueue_task(processed, complexity="high")
    return {
        "model_used": select_model("high").model_name,
        "simulation_output": result
    }
