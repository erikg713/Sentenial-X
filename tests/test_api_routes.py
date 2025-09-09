def test_health_route(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_wormgpt_route(api_client):
    response = api_client.post("/wormgpt", json={"prompt": "malicious"})
    assert response.status_code == 200
    assert "risk_score" in response.json()

def test_alert_route(api_client):
    response = api_client.post("/alerts", json={"type": "ransomware"})
    assert response.status_code == 200
    assert "message" in response.json()
