# Phase 8: API Authentication + HTTP-level Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-key API authentication to all REST routes except `/health` and `/slack/events`, and add HTTP-level TestClient tests covering auth enforcement and all REST routes.

**Architecture:** `make_verify_api_key(settings)` in `src/core/auth.py` returns a FastAPI dependency that reads `settings.api_key` at call time — empty string disables auth (dev/test), non-empty enforces `X-API-Key` header validation. Applied via `dependencies=[Depends(verify_api_key)]`. HTTP tests use `TestClient` with monkeypatched settings and mocked registry/orchestrator/scheduler.

**Tech Stack:** FastAPI (`Depends`, `Header`, `HTTPException`), `fastapi.testclient.TestClient`, `pytest` monkeypatch fixture, `unittest.mock.MagicMock`

---

## File Structure

**New files:**
- `src/core/auth.py` — `make_verify_api_key(settings)` factory; ~10 lines
- `tests/core/test_auth.py` — 4 unit tests for the auth dependency
- `tests/test_api.py` — 14 HTTP-level TestClient tests

**Modified files:**
- `src/core/config.py` — add `api_key: str = ""` to `Settings`
- `main.py` — import auth, create `verify_api_key`, add `Depends` to 8 routes

---

### Task 1: Settings.api_key + src/core/auth.py

**Files:**
- Modify: `src/core/config.py`
- Create: `src/core/auth.py`
- Create: `tests/core/test_auth.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_auth.py`:

```python
# tests/core/test_auth.py
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from src.core.auth import make_verify_api_key


def _settings(api_key=""):
    s = MagicMock()
    s.api_key = api_key
    return s


def test_correct_key_does_not_raise():
    verify = make_verify_api_key(_settings(api_key="secret"))
    verify(x_api_key="secret")  # must not raise


def test_wrong_key_raises_401():
    verify = make_verify_api_key(_settings(api_key="secret"))
    with pytest.raises(HTTPException) as exc_info:
        verify(x_api_key="wrong")
    assert exc_info.value.status_code == 401
    assert "Invalid" in exc_info.value.detail


def test_missing_key_raises_401():
    verify = make_verify_api_key(_settings(api_key="secret"))
    with pytest.raises(HTTPException) as exc_info:
        verify(x_api_key="")
    assert exc_info.value.status_code == 401


def test_empty_settings_key_disables_auth():
    verify = make_verify_api_key(_settings(api_key=""))
    verify(x_api_key="")          # must not raise
    verify(x_api_key="anything")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_auth.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.auth'`

- [ ] **Step 3: Add `api_key` to Settings**

In `src/core/config.py`, add `api_key: str = ""` after `clients_file`:

```python
class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str
    chromadb_persist_dir: str = ".chromadb"
    clients_file: str = "clients.json"
    api_key: str = ""

    # Jira settings
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "AI"

    # Email settings
    email_imap_host: str = ""
    email_imap_user: str = ""
    email_imap_password: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587

    model_config = SettingsConfigDict(env_file=".env")
```

- [ ] **Step 4: Create `src/core/auth.py`**

```python
# src/core/auth.py
from fastapi import Header, HTTPException


def make_verify_api_key(settings):
    def verify_api_key(x_api_key: str = Header(default="")):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return verify_api_key
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/core/test_auth.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/core/config.py src/core/auth.py tests/core/test_auth.py
git commit -m "feat: add Settings.api_key and make_verify_api_key dependency"
```

---

### Task 2: Wire auth into main.py (TDD — write HTTP auth tests first)

**Files:**
- Modify: `main.py`
- Create: `tests/test_api.py` (auth tests only in this task)

- [ ] **Step 1: Write the failing HTTP auth tests**

Create `tests/test_api.py` with only the three auth tests:

```python
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
```

- [ ] **Step 2: Run to verify auth tests fail**

```bash
.venv/bin/python -m pytest tests/test_api.py -v
```

Expected: `test_missing_key_returns_401` and `test_wrong_key_returns_401` FAIL (return 200/201 instead of 401 — auth not wired yet). `test_health_requires_no_auth` PASS.

- [ ] **Step 3: Add imports to main.py**

In `main.py`, update the first fastapi import line:

```python
from fastapi import FastAPI, Request, Depends
```

Add auth imports after the existing imports block (before `settings = Settings()`):

```python
from src.core.auth import make_verify_api_key
```

After `settings = Settings()`, add:

```python
verify_api_key = make_verify_api_key(settings)
```

- [ ] **Step 4: Apply auth to /clients routes**

Replace the four `/clients` route decorators:

```python
@app.post("/clients", status_code=201, dependencies=[Depends(verify_api_key)])
def create_client(config: ClientConfig):
    registry.upsert(config)
    return config.model_dump()


@app.get("/clients", dependencies=[Depends(verify_api_key)])
def list_clients():
    return {"clients": [c.model_dump() for c in registry.all()]}


@app.get("/clients/{client_id}", dependencies=[Depends(verify_api_key)])
def get_client(client_id: str):
    config = registry.get(client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return config.model_dump()


@app.delete("/clients/{client_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_client(client_id: str):
    registry.delete(client_id)
```

- [ ] **Step 5: Apply auth to /analyze and /jobs routes**

```python
@app.post("/analyze", dependencies=[Depends(verify_api_key)])
def analyze(body: AnalyzeBody):
    config = registry.get(body.request.client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    result = orchestrator.process(body.request, config, body.clarification_state)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"result": str(result)}


@app.post("/jobs", status_code=201, dependencies=[Depends(verify_api_key)])
def create_job(job: ScheduledJob):
    scheduler.add_job(job, on_complete=make_deliver())
    return {"job_id": job.job_id}


@app.get("/jobs", dependencies=[Depends(verify_api_key)])
def list_jobs():
    return {"jobs": [j.model_dump() for j in scheduler.list_jobs()]}


@app.delete("/jobs/{job_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_job(job_id: str):
    scheduler.remove_job(job_id)
```

**Do NOT add auth to `/health` or `/slack/events`.**

- [ ] **Step 6: Run auth tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_api.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Verify no regressions in full suite**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 224 passed (217 existing + 4 auth unit tests + 3 HTTP auth tests), 0 failures.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: wire API key auth to /clients, /analyze, /jobs; add HTTP auth tests"
```

---

### Task 3: HTTP tests for /clients, /analyze, /jobs routes

**Files:**
- Modify: `tests/test_api.py` (append 11 tests)

- [ ] **Step 1: Append route tests to `tests/test_api.py`**

Add the following after the existing auth tests in `tests/test_api.py`:

```python
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

def test_analyze_unknown_client_returns_404(client, monkeypatch):
    monkeypatch.setattr(main_module.registry, "get", lambda client_id: None)
    payload = {
        "request": {
            "channel": "slack", "sender_id": "u1", "sender_name": "User",
            "text": "Show revenue", "timestamp": "2026-05-19T00:00:00Z", "client_id": "c1",
        },
        "clarification_state": None,
    }
    resp = client.post("/analyze", json=payload, headers=HEADERS)
    assert resp.status_code == 404


def test_analyze_known_client_returns_200(client, monkeypatch):
    config = ClientConfig(
        client_id="c1", name="Test Co", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
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
    resp = client.post("/analyze", json=payload, headers=HEADERS)
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
```

- [ ] **Step 2: Run new tests**

```bash
.venv/bin/python -m pytest tests/test_api.py -v
```

Expected: 14 passed.

- [ ] **Step 3: Run full suite**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: ~231 passed, 0 failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_api.py
git commit -m "test: HTTP-level route tests for /clients, /analyze, /jobs"
```

---

## Self-Review

**Spec coverage:**
- ✅ `Settings.api_key: str = ""` — Task 1
- ✅ `src/core/auth.py` with `make_verify_api_key` — Task 1
- ✅ Auth disabled when `api_key == ""` — Task 1 (`test_empty_settings_key_disables_auth`)
- ✅ 401 for wrong/missing key — Task 1 (unit) + Task 2 (HTTP)
- ✅ `X-API-Key` header — Task 1 (Header parameter)
- ✅ Auth on `/clients/*`, `/analyze`, `/jobs/*` — Task 2
- ✅ `/health` excluded — verified by `test_health_requires_no_auth`
- ✅ `/slack/events` excluded — not modified in any task
- ✅ HTTP tests — auth (3), /clients (5), /analyze (2), /jobs (3) = 14 total — Task 2 + Task 3

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `make_verify_api_key(settings)` accepts any object with `.api_key`. In unit tests: `MagicMock`. In production: real `Settings` instance. Consistent.
- `monkeypatch.setattr(main_module.registry, "get", lambda client_id: None)` — `ClientRegistry.get(client_id)` signature matches. Consistent.
- `monkeypatch.setattr(main_module.orchestrator, "process", MagicMock(return_value=mock_resp))` — `Orchestrator.process(request, config, clarification_state)` called by the route handler. The mock ignores args and returns `mock_resp`. Consistent.
- `Response(request_id="r1", text="Analysis complete.")` — `charts=[]` (empty list) and `deck_pptx=None` are JSON-serializable. No bytes encoding issue. Consistent.
