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

// libs/lib/agent.ts
/**
 * Sentenial-X Lib Agent Module
 * Provides core agent utilities for monitoring, tracing, and telemetry.
 */

import { v4 as uuidv4 } from "uuid";

export type EventSeverity = "info" | "warning" | "critical" | "high";

export interface TraceEvent {
  id: string;
  source: string;
  eventType: string;
  severity: EventSeverity;
  data?: Record<string, any>;
  timestamp: string;
}

export class Agent {
  private events: TraceEvent[] = [];

  constructor(private historySize: number = 500) {}

  /**
   * Logs a new trace event.
   */
  logEvent(
    source: string,
    eventType: string,
    severity: EventSeverity = "info",
    data?: Record<string, any>
  ): string {
    const event: TraceEvent = {
      id: uuidv4(),
      source,
      eventType,
      severity,
      data,
      timestamp: new Date().toISOString(),
    };

    this.events.push(event);
    if (this.events.length > this.historySize) {
      this.events.shift(); // maintain history size
    }

    console.info(`[Agent] Event logged: ${JSON.stringify(event)}`);
    return event.id;
  }

  /**
   * Retrieves all events, optionally filtered by severity.
   */
  getEvents(severityFilter?: EventSeverity): TraceEvent[] {
    if (severityFilter) {
      return this.events.filter(
        (e) => e.severity.toLowerCase() === severityFilter.toLowerCase()
      );
    }
    return [...this.events];
  }

  /**
   * Clears all stored events.
   */
  clearEvents(): void {
    console.warn(`[Agent] Clearing all ${this.events.length} trace events`);
    this.events = [];
  }

  /**
   * Returns the last N events.
   */
  recentEvents(limit: number = 10): TraceEvent[] {
    return this.events.slice(-limit);
  }
}

// Example usage
if (require.main === module) {
  const agent = new Agent(100);
  agent.logEvent("Cortex", "ThreatDetected", "critical", { threat: "malware_xyz" });
  agent.logEvent("Orchestrator", "TaskCompleted", "info", { taskId: 1234 });

  console.log(agent.getEvents());
  console.log("Recent:", agent.recentEvents(1));
}
