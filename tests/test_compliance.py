from core.compliance.legal_ontology_parser import parse_policy

def test_gdpr_policy_detection():
    text = "All user data must be erasable upon request to comply with GDPR."
    result = parse_policy(text)
    assert "GDPR" in result["regulations"]
    assert result["action_required"] == "Data Erasure"

