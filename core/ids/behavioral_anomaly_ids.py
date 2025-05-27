# core/ids/behavioral_anomaly_ids.py

def run_detection(target):
    # Simulate behavior profile comparison
    baseline_behavior = {"requests_per_min": 10, "avg_payload_size": 512}
    current_behavior = {"requests_per_min": 58, "avg_payload_size": 1034}

    anomalies = []
    if current_behavior["requests_per_min"] > baseline_behavior["requests_per_min"] * 4:
        anomalies.append("High request rate")
    if current_behavior["avg_payload_size"] > baseline_behavior["avg_payload_size"] * 2:
        anomalies.append("Abnormal payload size")

    if anomalies:
        return {
            "status": "anomaly",
            "message": f"Behavioral anomalies: {', '.join(anomalies)}"
        }
    else:
        return {
            "status": "normal",
            "message": "No behavioral anomalies detected."
        }
