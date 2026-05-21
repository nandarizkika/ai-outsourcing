# Phase 10: Observability — Structured Logging + Request Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured JSON logging to stdout and per-request tracing via `X-Request-ID` header propagation through all log lines.

**Architecture:** A `ContextVar` stores the current `request_id` per request. A log filter injects it into every log record. `RequestLoggingMiddleware` generates or accepts `X-Request-ID`, sets the context var, and logs request start/end with timing. `configure_logging()` sets up a JSON handler on stdout globally at startup.

**Tech Stack:** Python `contextvars`, `python-json-logger>=2.0`, `starlette.middleware.base.BaseHTTPMiddleware`, stdlib `logging`, `fastapi.testclient.TestClient`, `pytest` `caplog`

---

## File Structure

**New files:**
- `src/core/request_context.py` — `ContextVar` + `get_request_id()` / `set_request_id()` helpers
- `src/core/logging_config.py` — `_RequestIdFilter` + `configure_logging(level="INFO")`
- `src/middleware/__init__.py` — empty package marker
- `src/middleware/logging.py` — `RequestLoggingMiddleware`
- `tests/core/test_request_context.py` — 3 unit tests
- `tests/core/test_logging_config.py` — 2 unit tests for `_RequestIdFilter`
- `tests/middleware/__init__.py` — empty package marker
- `tests/middleware/test_logging_middleware.py` — 4 HTTP tests

**Modified files:**
- `main.py` — import + call `configure_logging()`, add middleware
- `pyproject.toml` — add `python-json-logger>=2.0`

---

### Task 1: `src/core/request_context.py` (TDD)

**Files:**
- Create: `src/core/request_context.py`
- Create: `tests/core/test_request_context.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_request_context.py`:

```python
# tests/core/test_request_context.py
from src.core.request_context import get_request_id, set_request_id


def test_get_request_id_default_is_empty_string():
    set_request_id("")
    assert get_request_id() == ""


def test_set_and_get_request_id():
    set_request_id("abc123")
    assert get_request_id() == "abc123"


def test_request_id_isolated_per_context():
    from contextvars import copy_context
    set_request_id("outer")
    inner_value = []

    def run_in_copy():
        set_request_id("inner")
        inner_value.append(get_request_id())

    copy_context().run(run_in_copy)
    assert inner_value[0] == "inner"
    assert get_request_id() == "outer"
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_request_context.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.request_context'`

- [ ] **Step 3: Create `src/core/request_context.py`**

```python
# src/core/request_context.py
from contextvars import ContextVar

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)
```

- [ ] **Step 4: Run to verify 3 tests pass**

```bash
.venv/bin/python -m pytest tests/core/test_request_context.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -3
```

Expected: ~252 passed, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/core/request_context.py tests/core/test_request_context.py
git commit -m "feat: add request_context ContextVar for per-request tracing"
```

---

### Task 2: `src/core/logging_config.py` + dependency (TDD)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/core/logging_config.py`
- Create: `tests/core/test_logging_config.py`

- [ ] **Step 1: Add `python-json-logger` to `pyproject.toml`**

In `pyproject.toml`, add to the `dependencies` list:

```toml
"python-json-logger>=2.0",
```

- [ ] **Step 2: Install the dependency**

```bash
.venv/bin/pip install python-json-logger 2>&1 | tail -3
```

- [ ] **Step 3: Write the failing tests**

Create `tests/core/test_logging_config.py`:

```python
# tests/core/test_logging_config.py
import logging

from src.core.logging_config import _RequestIdFilter
from src.core.request_context import set_request_id


def test_filter_injects_request_id():
    set_request_id("test-req-id")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    f = _RequestIdFilter()
    f.filter(record)
    assert record.request_id == "test-req-id"


def test_filter_injects_empty_string_when_no_request():
    set_request_id("")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    f = _RequestIdFilter()
    f.filter(record)
    assert record.request_id == ""
```

- [ ] **Step 4: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_logging_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.logging_config'`

- [ ] **Step 5: Create `src/core/logging_config.py`**

```python
# src/core/logging_config.py
import logging
import sys

from pythonjsonlogger import jsonlogger

from src.core.request_context import get_request_id


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
```

- [ ] **Step 6: Run to verify 2 tests pass**

```bash
.venv/bin/python -m pytest tests/core/test_logging_config.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -3
```

Expected: ~254 passed, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/core/logging_config.py tests/core/test_logging_config.py
git commit -m "feat: add JSON logging config with request_id filter"
```

---

### Task 3: `src/middleware/logging.py` (TDD)

**Files:**
- Create: `src/middleware/__init__.py`
- Create: `src/middleware/logging.py`
- Create: `tests/middleware/__init__.py`
- Create: `tests/middleware/test_logging_middleware.py`

- [ ] **Step 1: Create package markers**

```bash
touch src/middleware/__init__.py tests/middleware/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/middleware/test_logging_middleware.py`:

```python
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
```

- [ ] **Step 3: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/middleware/test_logging_middleware.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.middleware.logging'` or tests fail because middleware isn't added yet.

- [ ] **Step 4: Create `src/middleware/logging.py`**

```python
# src/middleware/logging.py
import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.core.request_context import set_request_id

_logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
        set_request_id(request_id)

        _logger.info(
            "request started",
            extra={"method": request.method, "path": request.url.path},
        )

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000)

        _logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response
```

- [ ] **Step 5: Wire middleware into `main.py` temporarily to make tests pass**

In `main.py`, add these two imports near the top of the file (after existing imports):

```python
from src.core.logging_config import configure_logging
from src.middleware.logging import RequestLoggingMiddleware
```

Add `configure_logging()` call before the lifespan definition:

```python
configure_logging()
```

Add middleware after `app = FastAPI(lifespan=lifespan)`:

```python
app.add_middleware(RequestLoggingMiddleware)
```

- [ ] **Step 6: Run to verify middleware tests pass**

```bash
.venv/bin/python -m pytest tests/middleware/test_logging_middleware.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -3
```

Expected: ~257 passed, 0 failures.

- [ ] **Step 8: Commit all**

```bash
git add src/middleware/__init__.py src/middleware/logging.py tests/middleware/__init__.py tests/middleware/test_logging_middleware.py main.py
git commit -m "feat: add RequestLoggingMiddleware with X-Request-ID propagation and timing"
```

---

### Task 4: Verify full integration

**Files:**
- No new files — verify `main.py` changes are correct and full suite is green

- [ ] **Step 1: Verify `main.py` has all required changes**

```bash
grep -n "configure_logging\|RequestLoggingMiddleware\|add_middleware" main.py
```

Expected output should show:
- `from src.core.logging_config import configure_logging`
- `from src.middleware.logging import RequestLoggingMiddleware`
- `configure_logging()`
- `app.add_middleware(RequestLoggingMiddleware)`

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: ~257 passed, 0 failures.

- [ ] **Step 3: Smoke test the JSON logs manually**

```bash
.venv/bin/python -c "
from src.core.logging_config import configure_logging
from src.core.request_context import set_request_id
import logging
configure_logging()
set_request_id('test-123')
logging.getLogger('smoke').info('hello world')
"
```

Expected: a JSON line on stdout like:
```json
{"timestamp": "2026-05-21T...", "level": "INFO", "name": "smoke", "request_id": "test-123", "message": "hello world"}
```

- [ ] **Step 4: Commit if any final cleanups were needed**

Only commit if Step 1 revealed missing changes. Otherwise skip — Task 3 already committed everything.

```bash
git add main.py
git commit -m "feat: wire configure_logging and RequestLoggingMiddleware into main.py"
```

---

## Self-Review

**Spec coverage:**
- ✅ Structured JSON logging on stdout — Task 2 (`configure_logging`, `python-json-logger`)
- ✅ `request_id` in every log line — Task 2 (`_RequestIdFilter` + `ContextVar`)
- ✅ `X-Request-ID` header accepted when present — Task 3 (`test_passes_through_existing_request_id`)
- ✅ `X-Request-ID` generated when absent — Task 3 (`test_generates_request_id_when_absent`)
- ✅ `X-Request-ID` echoed in response — Task 3 (`response.headers["X-Request-ID"] = request_id`)
- ✅ Request start + end logged with timing — Task 3 (`test_logs_request_start_and_end`)
- ✅ Wired into `main.py` — Task 3 Step 5

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `set_request_id(str)` / `get_request_id() -> str` — used consistently in `_RequestIdFilter`, `RequestLoggingMiddleware`, and all tests.
- `uuid4().hex[:12]` — produces a 12-char lowercase hex string. `test_generates_request_id_when_absent` asserts `len == 12`. Consistent.
- `caplog.at_level(logging.INFO, logger="src.middleware.logging")` — matches `_logger = logging.getLogger(__name__)` in `src/middleware/logging.py` where `__name__` resolves to `src.middleware.logging`. Consistent.
