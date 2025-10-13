# tests/integration/test_cortex_integration.py
import pytest
import requests
from sentenialx.models.artifacts import register_artifact

def test_full_flow():
    # Register mock artifact
    mock_path = Path("sentenialx/models/artifacts/distill/mock.bin")
    mock_path.write_bytes(b"mock")
    register_artifact("distill", mock_path, "1.0.0")
    
    # Call API
    response = requests.post("http://localhost:8000/cortex/predict", json={"text": "Test threat"})
    assert response.status_code == 200
    assert "intent" in response.json()
