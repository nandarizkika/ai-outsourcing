# tests/test_api_phase11.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

import main as main_module
from main import app
from src.core.models import ClientConfig, Tier, Channel, Response

ADMIN_KEY = "test-admin-key"
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_key", ADMIN_KEY)
    with TestClient(app) as c:
        yield c


def _client_payload():
    return {
        "client_id": "c1", "name": "Test Co", "tier": "basic",
        "enabled_skills": [], "account_mode": "vendor",
        "active_channels": ["slack"],
    }


def test_create_client_returns_api_key(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "upsert", MagicMock())
    resp = client.post("/clients", json=_client_payload(), headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    data = resp.json()
    assert "api_key" in data
    assert len(data["api_key"]) == 64


def test_rotate_key_returns_new_key(client, monkeypatch):
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        api_key="old" * 16,
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    monkeypatch.setattr(main_module.registry, "upsert", MagicMock())
    resp = client.post("/clients/c1/rotate-key", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "api_key" in data
    assert len(data["api_key"]) == 64
    assert data["api_key"] != "old" * 16


def test_rotate_key_on_missing_client_returns_404(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: None)
    resp = client.post("/clients/nonexistent/rotate-key", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


def test_analyze_rejects_wrong_client_key(client, monkeypatch):
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        api_key="correctkey" * 6,
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    payload = {
        "request": {
            "channel": "slack", "sender_id": "u1", "sender_name": "User",
            "text": "Show revenue", "timestamp": "2026-05-19T00:00:00Z", "client_id": "c1",
        },
        "clarification_state": None,
    }
    resp = client.post(
        "/analyze", json=payload,
        headers={"X-Client-ID": "c1", "X-API-Key": "wrongkey"},
    )
    assert resp.status_code == 401


def test_analyze_accepts_correct_client_key(client, monkeypatch):
    client_key = "correctkey" * 6
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        api_key=client_key,
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    mock_resp = Response(request_id="r1", text="Analysis complete.")
    monkeypatch.setattr(main_module.orchestrator, "process", AsyncMock(return_value=mock_resp))
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


def test_analyze_rejects_mismatched_client_id(client, monkeypatch):
    client_key = "correctkey" * 6
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        api_key=client_key,
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    payload = {
        "request": {
            "channel": "slack", "sender_id": "u1", "sender_name": "User",
            "text": "Show revenue", "timestamp": "2026-05-19T00:00:00Z", "client_id": "c2",
        },
        "clarification_state": None,
    }
    resp = client.post(
        "/analyze", json=payload,
        headers={"X-Client-ID": "c1", "X-API-Key": client_key},
    )
    assert resp.status_code == 403


def test_rate_limiter_wired_to_app():
    from main import app, limiter
    assert app.state.limiter is limiter


def test_analyze_rate_limit_returns_429(client, monkeypatch):
    from limits import parse
    from slowapi.wrappers import Limit

    client_key = "correctkey" * 6
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        api_key=client_key,
    )
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: config)
    mock_resp = Response(request_id="r1", text="ok")
    monkeypatch.setattr(main_module.orchestrator, "process", AsyncMock(return_value=mock_resp))

    # Temporarily swap the route limit to 1/minute, then restore after test
    route_key = "main.analyze"
    original = main_module.limiter._route_limits.get(route_key, [])
    low_limit = Limit(
        limit=parse("1/minute"),
        key_func=main_module._get_client_id,
        scope=None, per_method=False, methods=None,
        error_message=None, exempt_when=None, cost=1, override_defaults=False,
    )
    main_module.limiter._route_limits[route_key] = [low_limit]
    main_module.limiter.reset()

    payload = {
        "request": {
            "channel": "slack", "sender_id": "u1", "sender_name": "User",
            "text": "hello", "timestamp": "2026-05-22T00:00:00Z", "client_id": "c1",
        },
        "clarification_state": None,
    }
    headers = {"X-Client-ID": "c1", "X-API-Key": client_key}

    try:
        resp1 = client.post("/analyze", json=payload, headers=headers)
        assert resp1.status_code == 200

        resp2 = client.post("/analyze", json=payload, headers=headers)
        assert resp2.status_code == 429
    finally:
        main_module.limiter._route_limits[route_key] = original
        main_module.limiter.reset()
