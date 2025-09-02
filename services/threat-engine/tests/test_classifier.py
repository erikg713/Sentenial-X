import os
from services.threat_engine.classifier import ThreatClassifier

def test_rule_match():
    clf = ThreatClassifier(
        model_path=os.path.join("services", "threat-engine", "models", "dummy_model.pkl"),
        rules_path=os.path.join("services", "threat-engine", "rules.yaml"),
    )
    result = clf.classify("DROP TABLE users;")
    assert result["threat"] is True
    assert result["method"] == "rule"
