# Telemetry

Sentenial-X streams all analysis results and simulator outputs into **TelemetryCollector**:

- Logs text analysis and feature analysis
- Collects anomaly detection results
- Triggers alerts and notifications

**Example:**

```json
{
  "text": "suspicious input",
  "report": {
    "text_analysis": {...},
    "anomaly": {...},
    "predictive": {...}
  }
}
