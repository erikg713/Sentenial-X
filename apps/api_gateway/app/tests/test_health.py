# app/tests/test_health.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "app" in r.json()


def test_health():
    r = client.get("/health/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "api-gateway"
    assert body["status"] == "ok"
    assert "uptime_seconds" in body