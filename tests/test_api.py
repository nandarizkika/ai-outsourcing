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


def _job_payload():
    return {
        "job_id": "j1", "client_id": "c1", "description": "Weekly report",
        "request_text": "Give me weekly summary",
        "cron_expression": "0 9 * * 1",
        "delivery_channel": "slack",
        "delivery_destination": "C123456",
    }


# --- /clients route tests ---

def test_post_client_creates_201(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "upsert", MagicMock())
    resp = client.post("/clients", json=_client_payload(), headers=HEADERS)
    assert resp.status_code == 201
    assert resp.json()["client_id"] == "c1"


def test_get_clients_returns_list(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "all", lambda: [])
    resp = client.get("/clients", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"clients": []}


def test_get_client_by_id_returns_200(client, monkeypatch):
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    resp = client.get("/clients/c1", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["client_id"] == "c1"


def test_get_missing_client_returns_404(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: None)
    resp = client.get("/clients/nonexistent", headers=HEADERS)
    assert resp.status_code == 404


def test_delete_client_returns_204(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "delete", MagicMock())
    resp = client.delete("/clients/c1", headers=HEADERS)
    assert resp.status_code == 204


# --- /analyze route tests ---

def test_analyze_wrong_credentials_returns_401(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: None)
    payload = {
        "request": {
            "channel": "slack", "sender_id": "u1", "sender_name": "User",
            "text": "Show revenue", "timestamp": "2026-05-19T00:00:00Z", "client_id": "c1",
        },
        "clarification_state": None,
    }
    resp = client.post("/analyze", json=payload, headers={"X-Client-ID": "c1", "X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_analyze_known_client_returns_200(client, monkeypatch):
    client_key = "test-client-key-" * 4
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        api_key=client_key,
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    mock_resp = Response(request_id="r1", text="Analysis complete.")
    monkeypatch.setattr(main_module.orchestrator, "process", MagicMock(return_value=mock_resp))
    payload = {
        "request": {
            "channel": "slack", "sender_id": "u1", "sender_name": "User",
            "text": "Show revenue", "timestamp": "2026-05-19T00:00:00Z", "client_id": "c1",
        },
        "clarification_state": None,
    }
    resp = client.post(
        "/analyze", json=payload,
        headers={"X-Client-ID": "c1", "X-API-Key": client_key},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Analysis complete."


# --- /jobs route tests ---

def test_post_job_returns_201(client, monkeypatch):
    monkeypatch.setattr(main_module.scheduler, "add_job", MagicMock())
    resp = client.post("/jobs", json=_job_payload(), headers=HEADERS)
    assert resp.status_code == 201
    assert resp.json()["job_id"] == "j1"


def test_get_jobs_returns_list(client, monkeypatch):
    monkeypatch.setattr(main_module.scheduler, "list_jobs", lambda: [])
    resp = client.get("/jobs", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"jobs": []}


def test_delete_job_returns_204(client, monkeypatch):
    monkeypatch.setattr(main_module.scheduler, "remove_job", MagicMock())
    resp = client.delete("/jobs/j1", headers=HEADERS)
    assert resp.status_code == 204
