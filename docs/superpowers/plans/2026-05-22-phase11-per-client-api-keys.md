# Phase 11: Per-Client API Keys Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single global API key on `/analyze` with per-client API keys — each client gets a unique key generated on creation, with a rotation endpoint for key refresh.

**Architecture:** `ClientConfig` gains an `api_key` field stored in `clients.json`. A new `make_verify_client_api_key(registry)` dependency reads `X-Client-ID` + `X-API-Key` headers and validates them against the registry. `/analyze` switches to this dependency; all `/clients` and `/jobs` management routes keep the existing global `verify_api_key`. Keys are generated with `secrets.token_hex(32)` (64-char hex) and returned once on creation or rotation.

**Tech Stack:** Python `secrets`, `fastapi.Header`, `fastapi.HTTPException`, `fastapi.testclient.TestClient`, `pytest` monkeypatch, `pydantic` `model_copy`

---

## File Structure

**Modified files:**
- `src/core/models.py` — add `api_key: Optional[str] = None` to `ClientConfig`
- `src/core/auth.py` — add `make_verify_client_api_key(registry)` factory
- `main.py` — add `import secrets`, import + wire `verify_client_api_key`, generate key in `POST /clients`, add `POST /clients/{client_id}/rotate-key`, swap `/analyze` dependency
- `tests/core/test_models_phase7.py` — 2 new model tests
- `tests/core/test_auth.py` — 3 new unit tests
- `tests/test_api.py` — update 2 existing `/analyze` tests to match new behavior

**New files:**
- `tests/test_api_phase11.py` — 5 new HTTP tests for key creation, rotation, and `/analyze` auth

---

### Task 1: `api_key` field on `ClientConfig` (TDD)

**Files:**
- Modify: `src/core/models.py`
- Modify: `tests/core/test_models_phase7.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_models_phase7.py` (append after the last test):

```python
def test_client_config_api_key_defaults_to_none():
    config = ClientConfig(**_base_config())
    assert config.api_key is None


def test_client_config_accepts_api_key():
    config = ClientConfig(**_base_config(api_key="abc123secret"))
    assert config.api_key == "abc123secret"
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_models_phase7.py -v 2>&1 | tail -10
```

Expected: `TypeError` — `ClientConfig.__init__` has no `api_key` parameter.

- [ ] **Step 3: Add `api_key` to `ClientConfig`**

In `src/core/models.py`, the `ClientConfig` class currently ends with:

```python
    connector_type: Optional[Literal["postgres", "mysql", "bigquery", "snowflake"]] = None
    connector_config: Optional[dict] = None
```

Add one line after `connector_config`:

```python
    connector_type: Optional[Literal["postgres", "mysql", "bigquery", "snowflake"]] = None
    connector_config: Optional[dict] = None
    api_key: Optional[str] = None
```

- [ ] **Step 4: Run to verify 2 tests pass**

```bash
.venv/bin/python -m pytest tests/core/test_models_phase7.py -v 2>&1 | tail -10
```

Expected: all tests in file pass (was 7, now 9).

- [ ] **Step 5: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -3
```

Expected: ~260 passed, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/core/models.py tests/core/test_models_phase7.py
git commit -m "feat: add api_key field to ClientConfig"
```

---

### Task 2: `make_verify_client_api_key` in `auth.py` (TDD)

**Files:**
- Modify: `src/core/auth.py`
- Modify: `tests/core/test_auth.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_auth.py` (append after existing tests):

```python
from src.core.auth import make_verify_client_api_key


def _make_registry(client=None):
    r = MagicMock()
    r.get.return_value = client
    return r


def _make_client_config(api_key="secret"):
    c = MagicMock()
    c.api_key = api_key
    return c


def test_verify_client_api_key_correct_key():
    registry = _make_registry(client=_make_client_config(api_key="secret"))
    verify = make_verify_client_api_key(registry)
    verify(x_client_id="c1", x_api_key="secret")  # must not raise


def test_verify_client_api_key_wrong_key_raises_401():
    registry = _make_registry(client=_make_client_config(api_key="secret"))
    verify = make_verify_client_api_key(registry)
    with pytest.raises(HTTPException) as exc_info:
        verify(x_client_id="c1", x_api_key="wrong")
    assert exc_info.value.status_code == 401
    assert "Invalid" in exc_info.value.detail


def test_verify_client_api_key_missing_client_raises_401():
    registry = _make_registry(client=None)
    verify = make_verify_client_api_key(registry)
    with pytest.raises(HTTPException) as exc_info:
        verify(x_client_id="unknown", x_api_key="anything")
    assert exc_info.value.status_code == 401
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_auth.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'make_verify_client_api_key'`.

- [ ] **Step 3: Implement `make_verify_client_api_key` in `src/core/auth.py`**

The full file after the change:

```python
from fastapi import Header, HTTPException


def make_verify_api_key(settings):
    def verify_api_key(x_api_key: str = Header(default="")):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return verify_api_key


def make_verify_client_api_key(registry):
    def verify_client_api_key(
        x_client_id: str = Header(default=""),
        x_api_key: str = Header(default=""),
    ):
        client = registry.get(x_client_id)
        if client is None or client.api_key != x_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return verify_client_api_key
```

- [ ] **Step 4: Run to verify 3 new tests pass**

```bash
.venv/bin/python -m pytest tests/core/test_auth.py -v 2>&1 | tail -15
```

Expected: all 7 tests in file pass (was 4, now 7).

- [ ] **Step 5: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -3
```

Expected: ~262 passed, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/core/auth.py tests/core/test_auth.py
git commit -m "feat: add make_verify_client_api_key dependency factory"
```

---

### Task 3: Wire into `main.py` + HTTP tests (TDD)

**Files:**
- Modify: `main.py`
- Create: `tests/test_api_phase11.py`
- Modify: `tests/test_api.py` (2 tests)

- [ ] **Step 1: Write the new HTTP tests**

Create `tests/test_api_phase11.py`:

```python
# tests/test_api_phase11.py
import pytest
from unittest.mock import MagicMock
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
```

- [ ] **Step 2: Update two broken tests in `tests/test_api.py`**

The `/analyze` route will switch from `verify_api_key` to `verify_client_api_key`, which changes behavior for the two existing `/analyze` tests. Update them:

**Find and replace `test_analyze_unknown_client_returns_404`** (currently raises 404 from route handler; after the change, the dependency raises 401 before the handler runs):

```python
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
```

**Find and replace `test_analyze_known_client_returns_200`** (must now send `X-Client-ID` and use a config with `api_key` set):

```python
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
```

- [ ] **Step 3: Run to verify new tests fail and updated old tests now expect correct behavior**

```bash
.venv/bin/python -m pytest tests/test_api_phase11.py tests/test_api.py -v 2>&1 | tail -20
```

Expected: all 5 new tests fail (feature not implemented); the 2 updated tests in test_api.py may also fail until implementation.

- [ ] **Step 4: Implement the changes in `main.py`**

**4a. Add `import secrets` and update the auth import** — change line 32 from:

```python
from src.core.auth import make_verify_api_key
```

to:

```python
import secrets
from src.core.auth import make_verify_api_key, make_verify_client_api_key
```

**4b. Add `verify_client_api_key` after registry is initialized** — after line 59 (`registry = ClientRegistry(settings.clients_file)`), add:

```python
registry = ClientRegistry(settings.clients_file)
verify_client_api_key = make_verify_client_api_key(registry)
```

**4c. Update `POST /clients` to generate and return a key** — change:

```python
@app.post("/clients", status_code=201, dependencies=[Depends(verify_api_key)])
def create_client(config: ClientConfig):
    registry.upsert(config)
    return config.model_dump()
```

to:

```python
@app.post("/clients", status_code=201, dependencies=[Depends(verify_api_key)])
def create_client(config: ClientConfig):
    api_key = secrets.token_hex(32)
    config = config.model_copy(update={"api_key": api_key})
    registry.upsert(config)
    return {"client_id": config.client_id, "api_key": api_key}
```

**4d. Add `POST /clients/{client_id}/rotate-key`** — add after the `DELETE /clients/{client_id}` route (currently at line ~183):

```python
@app.post("/clients/{client_id}/rotate-key", dependencies=[Depends(verify_api_key)])
def rotate_client_key(client_id: str):
    config = registry.get(client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    new_key = secrets.token_hex(32)
    registry.upsert(config.model_copy(update={"api_key": new_key}))
    return {"api_key": new_key}
```

**4e. Swap the `/analyze` dependency** — change:

```python
@app.post("/analyze", dependencies=[Depends(verify_api_key)])
```

to:

```python
@app.post("/analyze", dependencies=[Depends(verify_client_api_key)])
```

- [ ] **Step 5: Run new HTTP tests to verify 5 pass**

```bash
.venv/bin/python -m pytest tests/test_api_phase11.py -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 6: Run full suite — 0 failures**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: ~265 passed, 0 failures.

- [ ] **Step 7: Commit all**

```bash
git add main.py tests/test_api_phase11.py tests/test_api.py
git commit -m "feat: per-client API keys — generation, rotation, and /analyze auth"
```

---

## Self-Review

**Spec coverage:**
- ✅ `api_key: Optional[str] = None` on `ClientConfig` — Task 1
- ✅ Keys generated with `secrets.token_hex(32)` — Task 3 Step 4c
- ✅ Key returned in `POST /clients` response — Task 3 Step 4c + `test_create_client_returns_api_key`
- ✅ `POST /clients/{client_id}/rotate-key` — Task 3 Step 4d + `test_rotate_key_returns_new_key`
- ✅ Rotation 404 on missing client — `test_rotate_key_on_missing_client_returns_404`
- ✅ `make_verify_client_api_key` reads `X-Client-ID` + `X-API-Key` — Task 2
- ✅ `/analyze` uses `verify_client_api_key` — Task 3 Step 4e
- ✅ `/clients` and `/jobs` keep global `verify_api_key` — unchanged in Task 3
- ✅ Wrong key → 401 — `test_analyze_rejects_wrong_client_key`
- ✅ Correct key → 200 — `test_analyze_accepts_correct_client_key`

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `secrets.token_hex(32)` returns a 64-char lowercase hex string. All test assertions use `len == 64`. Consistent.
- `make_verify_client_api_key(registry)` — `registry` is the same `ClientRegistry` instance used throughout `main.py`. Consistent.
- `config.model_copy(update={"api_key": new_key})` — `model_copy` is the Pydantic v2 API already used in the codebase (e.g., `AnalyzeBody`). Consistent.
- `client.api_key != x_api_key` — both are `str`; `x_api_key` defaults to `""`. If `client.api_key` is `None` (pre-Phase-11 client), `None != ""` is `True` → raises 401. Correct: old clients are gated until provisioned via rotate-key. Consistent with spec.
