# Phase 10: Observability — Structured Logging + Request Tracing

**Date:** 2026-05-21

---

## Goal

Add structured JSON logging and per-request tracing to the FastAPI service. Every log line emits JSON to stdout with a consistent set of fields including `request_id`, enabling debugging, performance monitoring, and log aggregation.

---

## Architecture

Three components work together:

1. **JSON formatter** — `python-json-logger` wraps stdlib `logging` (already used in agents) to emit JSON lines on stdout. `configure_logging()` called once at startup wires it globally.

2. **Request context** — A `contextvars.ContextVar` stores the current `request_id` for each request. A log filter injects it into every log record automatically, so existing agent loggers gain `request_id` without code changes.

3. **Logging middleware** — `RequestLoggingMiddleware` generates or accepts an `X-Request-ID` header, sets the context var, times the request, and emits two structured log lines per request (start + end with `duration_ms` and `status_code`). The `X-Request-ID` is echoed back in the response.

---

## Log Line Format

Every log line is a JSON object on stdout:

```json
{
  "timestamp": "2026-05-21T07:00:00.123Z",
  "level": "INFO",
  "logger": "src.orchestrator.orchestrator",
  "request_id": "a1b2c3d4e5f6",
  "message": "request started",
  "method": "POST",
  "path": "/analyze",
  "client_id": "acme"
}
```

Request end line adds `status_code` and `duration_ms`:

```json
{
  "timestamp": "2026-05-21T07:00:01.456Z",
  "level": "INFO",
  "logger": "src.middleware.logging",
  "request_id": "a1b2c3d4e5f6",
  "message": "request completed",
  "method": "POST",
  "path": "/analyze",
  "status_code": 200,
  "duration_ms": 1333
}
```

---

## Component Designs

### 1. `src/core/request_context.py`

```python
from contextvars import ContextVar

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)
```

---

### 2. `src/core/logging_config.py`

```python
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
        fmt="%(timestamp)s %(level)s %(name)s %(request_id)s %(message)s",
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

---

### 3. `src/middleware/__init__.py`

Empty file — makes `src/middleware/` a package.

---

### 4. `src/middleware/logging.py`

```python
import time
import logging
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

---

### 5. `main.py` changes

Add before `app = FastAPI(lifespan=lifespan)`:

```python
from src.core.logging_config import configure_logging
from src.middleware.logging import RequestLoggingMiddleware

configure_logging()
```

After `app = FastAPI(lifespan=lifespan)`:

```python
app.add_middleware(RequestLoggingMiddleware)
```

---

### 6. `pyproject.toml`

Add to dependencies:

```toml
"python-json-logger>=2.0",
```

---

## Request ID Flow

```
Client → POST /analyze (X-Request-ID: abc123)
  → Middleware: reads "abc123", sets ContextVar, logs "request started"
    → Route handler → Orchestrator → Agents (all log with request_id="abc123")
  → Middleware: logs "request completed" with duration_ms + status_code
← Response (X-Request-ID: abc123 echoed back)

Client → GET /health (no X-Request-ID header)
  → Middleware: generates "f3a9b2c1d4e5", same flow
← Response (X-Request-ID: f3a9b2c1d4e5)
```

---

## File Structure

**New files:**
- `src/core/request_context.py` — ContextVar storage and helpers
- `src/core/logging_config.py` — JSON formatter + request_id filter + `configure_logging()`
- `src/middleware/__init__.py` — empty package marker
- `src/middleware/logging.py` — `RequestLoggingMiddleware`
- `tests/core/test_request_context.py` — unit tests for get/set helpers
- `tests/middleware/__init__.py` — empty package marker
- `tests/middleware/test_logging_middleware.py` — HTTP-level middleware tests

**Modified files:**
- `main.py` — import and call `configure_logging()`, add middleware
- `pyproject.toml` — add `python-json-logger>=2.0`

---

## Testing Strategy

**`tests/core/test_request_context.py`** (3 unit tests):
- `test_get_request_id_default_is_empty_string` — default when no context set
- `test_set_and_get_request_id` — set then get returns same value
- `test_request_id_isolated_per_context` — different threads/tasks get independent values

**`tests/middleware/test_logging_middleware.py`** (4 HTTP tests using `TestClient`):
- `test_generates_request_id_when_absent` — no header → response has `X-Request-ID`
- `test_passes_through_existing_request_id` — header present → same value echoed in response
- `test_request_id_in_response_header` — response always has `X-Request-ID`
- `test_logs_request_start_and_end` — caplog captures two log records per request with correct fields

---

## Self-Review

**Placeholder scan:** No TBDs or incomplete sections.

**Internal consistency:**
- `_RequestIdFilter` reads from the same `ContextVar` that `RequestLoggingMiddleware` writes — consistent.
- `configure_logging()` clears existing handlers before adding the JSON handler — prevents duplicate log lines if called multiple times or if pytest adds its own handlers.
- `uuid4().hex[:12]` produces a 12-char hex string — short enough to read in logs, unique enough for a single service instance.
- `set_request_id` uses `ContextVar.set()` which is async-safe — each request coroutine gets its own context copy in FastAPI's async environment.

**Scope check:** 2 new packages + middleware + context = one focused phase. No overlap with auth (Phase 8) or connectors (Phase 9).

**Ambiguity check:**
- Logging level: `INFO` default, configurable via `configure_logging(level=...)`. Production can pass `LOG_LEVEL` env var — not in scope for this phase but the hook is there.
- `/health` is included in middleware — it will log every load balancer probe. Acceptable for now; filtering it out is a follow-up if log volume becomes an issue.
