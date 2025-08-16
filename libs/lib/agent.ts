"""
Sentenial-X AI Module: Threat Analysis
--------------------------------------
Use GPT-4.1 via AI SDK to generate insights on logs, alerts, or suspicious activity.
"""

import asyncio
from ai import generate_text
from ai_sdk.openai import openai
from sentenial_core.logger import logger  # Sentenial-X logging system


async def analyze_threat(log_data: str, context: str = "cybersecurity incident") -> dict:
    """
    Analyze raw threat logs or alerts using GPT-4.1.
    
    Args:
        log_data (str): Raw logs, alerts, or suspicious activity to analyze.
        context (str): Context for the analysis (default: 'cybersecurity incident').
    
    Returns:
        dict: {'analysis': str, 'context': str}
    """
    prompt = f"Analyze the following {context} data and provide insights, severity, and recommendations:\n\n{log_data}"
    try:
        response = await generate_text(
            model=openai("gpt-4.1"),
            prompt=prompt
        )
        analysis = response.get("text", "No analysis returned by AI.")
        logger.info(f"Threat Analysis AI Response: {analysis}")
        return {"analysis": analysis, "context": context}
    except Exception as e:
        logger.error(f"Threat analysis failed: {e}")
        return {"analysis": None, "context": context, "error": str(e)}


# Example usage
if __name__ == "__main__":
    async def main():
        sample_logs = """
        Failed login attempts detected from IP 185.34.12.9
        Multiple privilege escalation attempts in /var/log/auth.log
        """
        result = await analyze_threat(sample_logs)
        print(result)

    asyncio.run(main())