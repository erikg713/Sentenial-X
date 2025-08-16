# apps/threat-engine/telemetry_analyzer.py
import pandas as pd

class TelemetryAnalyzer:
    """
    Analyzes telemetry metrics for anomalies.
    """

    def __init__(self, thresholds=None):
        # Example thresholds for CPU, memory, network anomalies
        self.thresholds = thresholds or {"cpu": 90, "memory": 90, "network": 1000}

    def detect_anomalies(self, telemetry_data):
        """
        Detect telemetry anomalies based on thresholds.
        telemetry_data: list of dicts with agent_id, metric_name, value
        """
        df = pd.DataFrame(telemetry_data)
        threats = []

        for metric, threshold in self.thresholds.items():
            if metric in df['metric_name'].unique():
                subset = df[df['metric_name'] == metric]
                anomalies = subset[subset['value'] > threshold]
                for _, row in anomalies.iterrows():
                    threats.append({
                        "type": f"{metric}_anomaly",
                        "agent_id": row['agent_id'],
                        "value": row['value']
                    })
        return threats 