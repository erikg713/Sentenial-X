# src/sentenial_x/ai/pipeline.py
import json
from sentenial_x.ai.engine import LLMEngine
from sentenial_x.ai.prompts import HTTP_ANALYSIS
from sentenial_x.ai.utils import serialize_session

class SessionAnalyzer:
    def __init__(self, model_name: str = "gpt-j-6B"):
        self.llm = LLMEngine(model_name)

    def analyze(self, session_obj):
        # 1. Serialize session for the model
        prompt = HTTP_ANALYSIS.format(
            session_json=serialize_session(session_obj)
        )

        # 2. Call the LLM
        raw_out = self.llm.generate(prompt, max_tokens=512, temperature=0.0)

        # 3. Parse and validate JSON
        try:
            result = json.loads(raw_out)
        except json.JSONDecodeError:
            # fallback: wrap raw output in minimal structure
            result = {"findings": [], "risk_score": 0, "remediation": raw_out}

        return result

