import asyncio
from typing import Dict, List, Any

from models.base import SemanticEvent, AnalysisResult
from processors.log_parser import LogParser
from services.ai_inference import run_inference
from services.risk_scoring import calculate_risk_score
from services.countermeasure_suggester import suggest_countermeasures
from utils.logger import init_logger

logger = init_logger("semantic_analyzer_pipeline")


class SemanticAnalyzerPipeline:
    """
    Production-ready async pipeline for threat event processing:
    1. Parse raw events
    2. Run AI analysis
    3. Calculate risk scores
    4. Suggest countermeasures
    """

    def __init__(self):
        self.parser = LogParser()

    async def process_event(self, raw_log: str, source: str) -> Dict[str, Any]:
        # Step 1: Parse log
        event: SemanticEvent = self.parser.parse_log(raw_log, source)
        logger.info(f"Parsed Event: {event.event_id}")

        # Step 2: Run AI inference
        analysis: AnalysisResult = await run_inference(event.dict())
        logger.info(f"AI Analysis Complete: {analysis.event_id}")

        # Step 3: Calculate risk
        risk = calculate_risk_score(analysis.severity, analysis.risk_score)
        logger.info(f"Calculated Risk Score: {risk} for event {analysis.event_id}")

        # Step 4: Suggest countermeasures
        countermeasures = suggest_countermeasures(analysis)
        logger.info(f"Suggested Countermeasures: {countermeasures} for event {analysis.event_id}")

        # Step 5: Return structured pipeline result
        return {
            "event_id": event.event_id,
            "source": event.source,
            "timestamp": event.timestamp,
            "analysis": analysis.summary,
            "severity": analysis.severity,
            "risk_score": risk,
            "countermeasures": countermeasures["countermeasures"],
        }

    async def process_batch(self, raw_logs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of raw logs asynchronously.
        Each log item: {"log": str, "source": str}
        """
        tasks = [self.process_event(item["log"], item["source"]) for item in raw_logs]
        results = await asyncio.gather(*tasks)
        logger.info(f"Processed {len(results)} events in batch")
        return results


# -------------------------------
# Example usage / live test
# -------------------------------
if __name__ == "__main__":
    async def main():
        pipeline = SemanticAnalyzerPipeline()
        sample_logs = [
            {"log": "Failed login attempts from IP 185.34.12.9", "source": "AuthService"},
            {"log": "Multiple privilege escalation attempts detected", "source": "Cortex"},
            {"log": "RCE attempt in /var/www/html/index.php", "source": "WebServer"},
        ]

        results = await pipeline.process_batch(sample_logs)
        for res in results:
            print(res)

    asyncio.run(main())
