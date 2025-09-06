# tests/test_api.py
import pytest
import json
from httpx import AsyncClient
from apps.api import app  # Adjust import if your FastAPI/Flask app is elsewhere

@pytest.mark.asyncio
async def test_healthcheck():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_threat_feed_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/threats/latest")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            threat = data[0]
            assert "ip" in threat
            assert "type" in threat
            assert "severity" in threat

@pytest.mark.asyncio
async def test_exploit_trigger_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"target": "192.168.1.100", "exploit": "ms17_010_eternalblue"}
        response = await client.post("/exploit/run", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "output" in result
        assert isinstance(result["output"], str)

@pytest.mark.asyncio
async def test_invalid_exploit():
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"target": "192.168.1.100", "exploit": "nonexistent_module"}
        response = await client.post("/exploit/run", json=payload)
        assert response.status_code == 400
        result = response.json()
        assert "error" in result
