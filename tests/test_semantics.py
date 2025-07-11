import pytest
from core.engine.semantics_analyzer import analyze_request

def test_injection_detection():
    mock_payload = {
        "headers": {"User-Agent": "curl"},
        "body": "id=1 OR 1=1",
        "query": "id=1 OR 1=1"
    }
    result = analyze_request(mock_payload)
    assert result["intent"] == "SQLInjection"
    assert result["severity"] == "high"

