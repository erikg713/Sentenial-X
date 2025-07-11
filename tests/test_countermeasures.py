from core.countermeasures.rules_engine import apply_rules

def test_trigger_sandbox():
    threat_data = {
        "intent": "CommandInjection",
        "severity": "high",
    }
    result = apply_rules(threat_data)
    assert result["action"] == "sandbox"

