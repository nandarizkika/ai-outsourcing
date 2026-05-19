# tests/test_api.py
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import main as main_module
from main import app
from src.core.models import ClientConfig, Tier, Channel, Response

TEST_KEY = "test-secret-key"
HEADERS = {"X-API-Key": TEST_KEY}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_key", TEST_KEY)
    with TestClient(app) as c:
        yield c


def _client_payload():
    return {
        "client_id": "c1", "name": "Test Co", "tier": "basic",
        "enabled_skills": [], "account_mode": "vendor",
        "active_channels": ["slack"],
    }


# --- Auth tests ---

def test_missing_key_returns_401(client):
    resp = client.post("/clients", json=_client_payload())
    assert resp.status_code == 401


def test_wrong_key_returns_401(client):
    resp = client.post("/clients", json=_client_payload(), headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_health_requires_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
