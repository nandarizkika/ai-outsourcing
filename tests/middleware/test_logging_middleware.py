# tests/middleware/test_logging_middleware.py
import logging
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_generates_request_id_when_absent(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) == 12


def test_passes_through_existing_request_id(client):
    resp = client.get("/health", headers={"X-Request-ID": "myid12345678"})
    assert resp.headers["x-request-id"] == "myid12345678"


def test_request_id_present_on_all_responses(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    resp2 = client.get("/health", headers={"X-Request-ID": "custom-id-123"})
    assert resp2.headers["x-request-id"] == "custom-id-123"


def test_logs_request_start_and_end(client, caplog):
    with caplog.at_level(logging.INFO, logger="src.middleware.logging"):
        client.get("/health")
    messages = [r.message for r in caplog.records]
    assert any("request started" in m for m in messages)
    assert any("request completed" in m for m in messages)
