# Phase 8: API Authentication + HTTP-level Tests

**Date:** 2026-05-19

---

## Goal

Add a single-key API authentication layer to all REST endpoints (except `/health` and `/slack/events`), and introduce HTTP-level TestClient tests covering all REST routes.

---

## Architecture

Two parts:

1. **Auth layer** — a FastAPI `Depends` dependency in `src/core/auth.py` that reads `X-API-Key` from the request header and compares it against `settings.api_key`. Applied to 7 routes via `dependencies=[Depends(verify_api_key)]`. When `settings.api_key == ""` (default), auth is disabled — safe for local dev without an env file.

2. **HTTP tests** — `tests/test_api.py` using FastAPI `TestClient`. Two pytest fixtures cover the authenticated and unauthenticated caller scenarios. Registry, orchestrator, and scheduler are mocked so tests are fast and isolated.

---

## Component Designs

### 1. `Settings.api_key` (`src/core/config.py`)

Add one field:

```python
api_key: str = ""
```

Empty string = auth disabled. Non-empty = auth enforced.

---

### 2. `src/core/auth.py`

```python
from fastapi import Header, HTTPException
from src.core.config import Settings


def make_verify_api_key(settings: Settings):
    def verify_api_key(x_api_key: str = Header(default="")):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return verify_api_key
```

- `make_verify_api_key(settings)` is called once at startup in `main.py`, returning the dependency.
- The closure reads `settings.api_key` at call time, so monkeypatching `settings.api_key` in tests works without reimporting.
- When `settings.api_key == ""`, the guard is skipped — no 401.

---

### 3. `main.py` changes

Add import:
```python
from src.core.auth import make_verify_api_key
from fastapi import Depends
```

After `settings = Settings()`, create the dependency:
```python
verify_api_key = make_verify_api_key(settings)
```

Apply to 7 routes (add `dependencies=[Depends(verify_api_key)]`):
- `POST /clients`
- `GET /clients`
- `GET /clients/{client_id}`
- `DELETE /clients/{client_id}`
- `POST /analyze`
- `POST /jobs`
- `GET /jobs`
- `DELETE /jobs/{job_id}`

**Excluded routes:**
- `GET /health` — load balancer probes don't carry credentials
- `POST /slack/events` — protected by Slack signing secret (HMAC-SHA256), not API key

---

### 4. `tests/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
import main as main_module
from main import app

TEST_KEY = "test-secret-key"


@pytest.fixture
def authed_client(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_key", TEST_KEY)
    with TestClient(app) as c:
        yield c, {"X-API-Key": TEST_KEY}


@pytest.fixture
def unauthed_client(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_key", TEST_KEY)
    with TestClient(app) as c:
        yield c
```

**Auth tests (using `unauthed_client`):**
- `test_missing_key_returns_401` — POST /clients with no header → 401
- `test_wrong_key_returns_401` — POST /clients with wrong key → 401
- `test_health_no_auth_required` — GET /health with no key → 200

**`/clients` tests (using `authed_client`, mock registry):**
- `test_post_client_creates_201`
- `test_get_clients_returns_list`
- `test_get_client_by_id_returns_200`
- `test_get_missing_client_returns_404`
- `test_delete_client_returns_204`

**`/analyze` tests (using `authed_client`, mock orchestrator):**
- `test_analyze_unknown_client_returns_404`
- `test_analyze_known_client_returns_200`

**`/jobs` tests (using `authed_client`, mock scheduler):**
- `test_post_job_returns_201`
- `test_get_jobs_returns_list`
- `test_delete_job_returns_204`

Registry, orchestrator, and scheduler are patched via `unittest.mock.patch` targeting the module-level objects in `main.py` (e.g., `patch("main.registry")`).

---

## File Structure

**New files:**
- `src/core/auth.py`
- `tests/test_api.py`

**Modified files:**
- `src/core/config.py` — add `api_key: str = ""`
- `main.py` — import auth, create `verify_api_key`, apply `Depends` to 7 routes

---

## Testing Strategy

All tasks follow TDD: write failing test → verify fail → implement → verify pass → commit.

**Expected test count:** 217 + ~14 new = ~231 tests.

---

## Self-Review

**Placeholder scan:** No TBDs or incomplete sections.

**Internal consistency:**
- `make_verify_api_key(settings)` closes over the settings object reference, so `monkeypatch.setattr(main_module.settings, "api_key", TEST_KEY)` affects the running dependency without reimport. Consistent with how Pydantic v2 models allow attribute mutation by default (not frozen).
- `/health` and `/slack/events` are explicitly excluded — documented in both the component design and the route list.
- The `default=""` in `Header(default="")` ensures requests with no `X-API-Key` header don't raise a FastAPI validation error before reaching the guard — the guard itself returns 401. This prevents a 422 Unprocessable Entity leaking instead of 401.

**Scope check:** 2 new files, 2 modified — appropriate for a single phase.

**Ambiguity check:**
- Auth disabled when `api_key == ""`: explicit. Prod deployments must set `API_KEY` in `.env`.
- Mock strategy: `patch("main.registry")` etc. targets module-level objects, which is the correct patch target for FastAPI route functions that close over them at definition time.
