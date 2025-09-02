import asyncio
from typing import Dict, Any
from models.base import AnalysisResult
from libs.ai.analyze_threat import analyze_threat
from utils.logger import init_logger

logger = init_logger("semantic_analyzer_ai")

async def run_inference(event: Dict[str, Any]) -> AnalysisResult:
    """
    Sends event data to AI for semantic analysis.
    """
    try:
        ai_output = await analyze_threat(str(event))
        return AnalysisResult(
            event_id=event["event_id"],
            severity=ai_output.get("severity", "medium"),
            risk_score=ai_output.get("risk_score", 0.5),
            summary=ai_output.get("analysis", "No summary provided"),
            recommendations=ai_output.get("recommendations", {})
        )
    except Exception as e:
        logger.error(f"AI inference failed for {event['event_id']}: {e}")
        return AnalysisResult(
            event_id=event["event_id"],
            severity="unknown",
            risk_score=0.0,
            summary="Inference failed",
            recommendations={}
        )
