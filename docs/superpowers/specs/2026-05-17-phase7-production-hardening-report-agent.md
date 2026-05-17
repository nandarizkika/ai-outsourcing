# Phase 7: Production Hardening + ReportAgent Design

**Date:** 2026-05-17

---

## Goal

Wire the SQLAgent to real per-client databases via a `ClientRegistry`, expose a `/analyze` HTTP endpoint and `/clients` CRUD API, build a `ReportAgent` that produces Markdown (Slack) and HTML (email) reports, and fix scheduled job delivery to actually send results through the appropriate channel.

---

## Architecture

Four tightly coupled parts:

1. **`ClientRegistry`** — owns `clients.json`, lazy-creates `SQLConnector` per client, backs the REST API.
2. **SQL wiring** — Orchestrator uses registry to create per-request `SQLAgent`. `SQLAgent.store` becomes optional.
3. **REST API** — `/clients` CRUD + `/analyze` endpoint added to `main.py`.
4. **`ReportAgent` + delivery** — formats analysis into Markdown/HTML; scheduled jobs deliver via Slack or Email.

All implementation follows TDD: write failing test → verify fail → implement → verify pass → commit.

---

## Data Model Changes

### `src/core/models.py`

Add to `ClientConfig`:
```python
database_url: Optional[str] = None
```

Add to `Response`:
```python
report_markdown: Optional[str] = None
report_html: Optional[str] = None
```

### `src/core/config.py`

Add to `Settings`:
```python
clients_file: str = "clients.json"
```

---

## Component Designs

### 1. `ClientRegistry` (`src/core/client_registry.py`)

Single class, ~80 lines. Owns a JSON file on disk and a lazy connector cache.

```python
class ClientRegistry:
    def __init__(self, filepath: str) -> None: ...
    def get(self, client_id: str) -> ClientConfig | None: ...
    def upsert(self, config: ClientConfig) -> None: ...
    def delete(self, client_id: str) -> None: ...
    def all(self) -> list[ClientConfig]: ...
    def get_connector(self, client_id: str) -> SQLConnector | None: ...
```

- `__init__`: creates the JSON file if missing (writes `{}`).
- `upsert`: serializes `ClientConfig` to dict, writes the whole file atomically.
- `delete`: removes from file and evicts `_connectors` cache.
- `get_connector`: if `config.database_url` is set and no cached connector exists for `client_id`, creates `SQLConnector(config.database_url)` and caches it. Returns `None` if no `database_url`.
- Thread safety: file writes use a simple `threading.Lock`. Sufficient for single-process FastAPI with default workers.

### 2. SQL Wiring — `SQLAgent` + `Orchestrator`

**`src/agents/sql_agent.py`** — make `store` optional:
```python
def __init__(self, llm: LLMRouter, connector: SQLConnector, store=None) -> None:
```
Skip `store.add()` when `store is None`. No other changes.

**`src/orchestrator/orchestrator.py`** — add registry support:

New import:
```python
from src.core.client_registry import ClientRegistry
```

New `__init__` parameter (after `ab_agent`):
```python
registry: ClientRegistry | None = None,
```

New `__init__` body line:
```python
self._registry = registry
```

In `process()`, replace the SQL block's agent lookup with:
```python
sql_agent = self._sql_agent
if self._registry is not None:
    connector = self._registry.get_connector(config.client_id)
    if connector is not None:
        sql_agent = SQLAgent(llm=self._llm, connector=connector)
```

Then use `sql_agent.run(...)` instead of `self._sql_agent.run(...)`. Fully backward-compatible — existing tests pass `sql_agent` mock directly and still work.

### 3. REST API (`main.py`)

Instantiate registry at startup:
```python
registry = ClientRegistry(settings.clients_file)
```

Pass to Orchestrator:
```python
orchestrator = Orchestrator(..., registry=registry)
```

**`/clients` endpoints** (mirrors `/jobs` pattern):
```
POST   /clients                 — upsert ClientConfig (201)
GET    /clients                 — list all configs
GET    /clients/{client_id}     — get one config (404 if missing)
DELETE /clients/{client_id}     — delete (204)
```

**`/analyze` endpoint**:
```
POST /analyze
Body: {"request": {...Request fields...}, "clarification_state": {...} | null}
Returns: Response | ClarificationState | AnalystResult (as JSON)
```

Implementation:
1. Parse `request.client_id` from body.
2. Look up config via `registry.get(client_id)`. Return 404 if not found.
3. Call `orchestrator.process(request, config, clarification_state)`.
4. Return result serialized as JSON.

**Slack/Email channel `client_configs` integration**: Replace the `client_configs = {}` dict with a thin adapter that delegates to the registry:
```python
class RegistryAdapter(dict):
    def __getitem__(self, key):
        config = registry.get(key)
        if config is None:
            raise KeyError(key)
        return config
    def __contains__(self, key):
        return registry.get(key) is not None
```
Pass `RegistryAdapter()` to `SlackChannel` and `EmailChannel` instead of the raw dict. No changes to channel classes.

### 4. `ReportAgent` (`src/agents/report_agent.py`)

```python
class ReportAgent:
    def __init__(self, llm=None) -> None: ...
    def run(self, client_id: str, request: str, data: dict) -> AgentResult: ...
```

`data` dict keys:
- `analysis_text: str` — the generated analysis from Orchestrator
- `sql_rows: list[dict]` — first 10 rows from sql_data (empty list if no SQL ran)
- `anomalies: list[dict]` — anomaly dicts (empty list if none)

**Implementation**:
1. LLM call (`TaskType.REASONING`) with system prompt instructing structured Markdown output:
   - `## Executive Summary` — 3-5 bullet points
   - `## Key Findings` — table of metrics if sql_rows present
   - `## Anomalies` — listed if anomalies non-empty, omitted if empty
   - `## Recommendation` — 1-2 sentence action
2. Convert Markdown to HTML using the `markdown` Python library (`pip install markdown`).
3. Return `AgentResult(success=True, data={"markdown": str, "html": str})`.
4. On LLM failure: return `AgentResult(success=False, error=str(exc))`.

**Orchestrator wiring** — add `report_agent: ReportAgent | None = None` parameter. After `_generate_response()` and before the Deck block:

```python
if (
    plan.get("report")
    and self._report_agent is not None
    and SkillModule.REPORT_GENERATION in config.enabled_skills
):
    report_result = self._report_agent.run(
        config.client_id,
        request.text,
        {
            "analysis_text": text,
            "sql_rows": (sql_data.get("rows", []) or [])[:10] if sql_data else [],
            "anomalies": [a.model_dump() for a in (anomalies or [])],
        },
    )
    if report_result.success:
        response_report_markdown = report_result.data["markdown"]
        response_report_html = report_result.data["html"]
```

`_plan()` system prompt gains:
```
"Set report=true for generate report/weekly summary/send me a report requests."
```

Fallback dict gains `"report": False`.

`Response` gets populated with `report_markdown` and `report_html` when report runs.

### 5. Scheduled Delivery

**`_noop_delivery` → `deliver(job, response)`** in `main.py`:

```python
def deliver(job: ScheduledJob, response: Response) -> None:
    content_md = response.report_markdown or response.text
    content_html = response.report_html or f"<pre>{response.text}</pre>"
    if job.delivery_channel == Channel.SLACK:
        slack.send_message(job.delivery_destination, content_md)
    elif job.delivery_channel == Channel.EMAIL:
        email_channel.send_email(job.delivery_destination, job.description, content_html)
```

Replace `_noop_delivery` with `deliver` in all `scheduler.add_job(job, on_complete=deliver)` calls.

Slack and Email channel classes need a `send_message` / `send_email` method added if not already present. Check existing implementations and add thin send wrappers if missing.

---

## File Structure

**New files:**
- `src/core/client_registry.py`
- `src/agents/report_agent.py`
- `tests/core/test_client_registry.py`
- `tests/agents/test_report_agent.py`
- `tests/test_integration_phase7.py`

**Modified files:**
- `src/core/models.py` — `ClientConfig.database_url`, `Response.report_markdown`, `Response.report_html`
- `src/core/config.py` — `Settings.clients_file`
- `src/agents/sql_agent.py` — `store` optional
- `src/orchestrator/orchestrator.py` — `registry` param, dynamic SQLAgent, `report_agent` param, report routing block, `_plan()` prompt + fallback
- `main.py` — registry instantiation, `RegistryAdapter`, `/clients` + `/analyze` routes, `report_agent` instantiation, `deliver` function

---

## Testing Strategy

All tasks follow TDD: write failing test → verify fail → implement → verify pass → commit.

### `tests/core/test_client_registry.py`
- `test_upsert_and_get_returns_config` — upsert then get returns same config
- `test_get_missing_returns_none` — unknown client_id returns None
- `test_delete_removes_config` — upsert then delete then get returns None
- `test_all_returns_all_configs` — list returns all upserted configs
- `test_persistence_across_instances` — two registry instances at same path share data
- `test_get_connector_returns_none_without_database_url` — config without database_url → None
- `test_get_connector_caches_instance` — two calls return same SQLConnector object
- `test_delete_evicts_connector_cache` — delete then get_connector creates fresh connector

### `tests/agents/test_report_agent.py`
- `test_report_returns_markdown_and_html` — mock LLM → both keys in data
- `test_html_contains_html_tag` — confirming markdown→HTML conversion ran
- `test_empty_sql_rows_still_produces_report` — graceful with no data
- `test_anomalies_included_in_prompt` — mock LLM call captures system prompt containing anomaly text
- `test_llm_failure_returns_error` — LLM raises exception → success=False

### `tests/test_integration_phase7.py`
- `test_post_clients_creates_client` — POST /clients → 201, retrievable via GET
- `test_get_clients_returns_all` — list endpoint
- `test_delete_client_removes_it` — DELETE → GET returns 404
- `test_analyze_unknown_client_returns_404` — POST /analyze with bad client_id
- `test_analyze_valid_client_returns_response` — POST /analyze with known client
- `test_report_agent_called_when_skill_enabled` — mock report_agent, assert called once
- `test_sql_agent_store_optional` — SQLAgent with store=None, successful query, store.add not called

**Expected total**: 189 + ~20 new = ~209 tests.

---

## Self-Review

**Placeholder scan:** No TBDs or incomplete sections.

**Internal consistency:**
- `RegistryAdapter` bridges the registry to Slack/Email channel without modifying channel classes — consistent with "follow existing patterns, don't refactor unrelated code."
- `SQLAgent.store=None` is backward-compatible; all existing tests pass a store mock and still work.
- `report_agent` follows the exact same optional-param pattern as all other agents in Orchestrator.
- `_plan()` fallback dict updated for `"report"` key — consistent with how `"hypothesis"`, `"segment"`, `"ab_test"` were added in Phase 6.

**Scope check:** 7-8 implementation tasks — appropriate for a single phase.

**Ambiguity check:**
- Delivery fallback: if `report_markdown` is None (report skill not enabled), `deliver()` uses `response.text`. Explicit.
- `RegistryAdapter` only needs `__getitem__` and `__contains__` — only what Slack/Email channels actually call on `client_configs`. Confirmed by reading channel code.
- `markdown` library: add `markdown` to `requirements.txt` / `pyproject.toml` as a new dependency.
