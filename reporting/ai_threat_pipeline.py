"""
Sentenial-X AI → Reporting Pipeline
-----------------------------------
Automatically analyze logs with GPT-4.1 and generate structured
reports ready to upload via the Sentenial-X reporting system.
"""

import asyncio
from ai import generate_text
from ai_sdk.openai import openai
from sentenial_core.logger import logger
from reporting import ReportBuilder, ReportUploader


async def analyze_and_report(log_data: str, context: str = "cybersecurity incident") -> dict:
    """
    Analyze raw threat logs and generate a Sentenial-X report.
    
    Args:
        log_data (str): Raw logs, alerts, or suspicious activity.
        context (str): Context for the analysis.
    
    Returns:
        dict: {'report_path': str, 'remote_uploaded': bool, 'analysis': str}
    """
    prompt = f"Analyze the following {context} and provide detailed insights, severity, and recommendations:\n\n{log_data}"

    # 1️⃣ Generate AI analysis
    try:
        ai_response = await generate_text(
            model=openai("gpt-4.1"),
            prompt=prompt
        )
        analysis_text = ai_response.get("text", "No analysis returned by AI.")
        logger.info(f"AI Threat Analysis: {analysis_text}")
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        analysis_text = f"Analysis failed: {e}"

    # 2️⃣ Build a structured report
    report = ReportBuilder(title="AI Threat Analysis Report")
    report.add_section("Context", context, severity="info")
    report.add_section("Raw Logs", log_data, severity="high")
    report.add_section("AI Analysis", analysis_text, severity="critical")

    # 3️⃣ Save & optionally upload the report
    uploader = ReportUploader(upload_dir="reports/ai")
    result = await uploader.upload(
        filename="ai_threat_report.txt",
        content=report.to_text(),  # or report.to_json() / report.to_markdown()
        remote_url=None  # can be set to your remote dashboard/api endpoint
    )

    # Include AI analysis in the return dict
    result["analysis"] = analysis_text
    return result


# Example usage
if __name__ == "__main__":
    async def main():
        sample_logs = """
        Failed login attempts detected from IP 185.34.12.9
        Multiple privilege escalation attempts in /var/log/auth.log
        Suspicious script execution detected in /tmp/malware.sh
        """
        report_result = await analyze_and_report(sample_logs)
        print(report_result)

    asyncio.run(main())