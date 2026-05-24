# DAG-Based Parallel Agent Execution Design

## Goals

Reduce `/analyze` response latency and make the agent dependency structure explicit. Agents that are independent run concurrently; agents that depend on prior results wait only for what they actually need.

## Architecture

`orchestrator.process()` becomes `async def process()`. Agents execute in 4 sequential stages; within each stage, independent agents run concurrently via `asyncio.gather()` + `asyncio.to_thread()`.

```
Stage 1 — Plan (sequential)
  LLM call: determines which agents to run
         ↓
Stage 2 — Data (parallel)
  [sql_agent, spreadsheet_agent]
  → produces sql_data
         ↓
Stage 3 — Analysis (parallel)
  [chart, anomaly, ml, hypothesis, segment, ab_test]
  → each reads sql_data; enrichment results merged after gather
         ↓
Stage 3b — Generate response (sequential)
  LLM call on enriched_data
         ↓
Stage 4 — Output (parallel)
  [report_agent, deck_agent]
         ↓
  Return Response
```

## Key Design Decisions

### Agent signatures unchanged
All 11 agent `.run()` methods stay synchronous. The orchestrator wraps them in `asyncio.to_thread()` — no agent files touched.

### sql_data mutation fix
`hypothesis_agent`, `segmentation_agent`, and `ab_agent` currently mutate `sql_data` in-place with `sql_data = {**(sql_data or {}), **result.data}`. In parallel execution this is a race condition. Fix: each agent's result dict is collected separately after `gather()` completes and merged into `enriched_data` before calling `_generate_response()`. No change to agent return values — just how the orchestrator handles them.

### `asyncio.gather(return_exceptions=True)`
All parallel stages use `return_exceptions=True`. If one agent in a stage fails, the others complete normally. After gathering, the orchestrator checks each result: exceptions are logged and skipped. A failed chart agent means no chart in the response — same behavior as the current `if result.success:` guards.

### `/analyze` endpoint goes async
`main.py`'s `analyze()` handler becomes `async def` and `await`s `orchestrator.process()`. No other routes change.

### Scheduled jobs
Any `orchestrator.process()` calls in `src/jobs/` are wrapped with `asyncio.run()` (or `asyncio.get_event_loop().run_until_complete()` if already in an async context).

## File Structure

| File | Change |
|---|---|
| `src/orchestrator/orchestrator.py` | `async def process()`, `_run_stage()` helper, 4-stage parallel execution |
| `main.py` | `async def analyze()`, `await orchestrator.process()` |
| `src/jobs/*.py` | Wrap `process()` calls with `asyncio.run()` if any exist |
| `tests/test_orchestrator.py` | Add `@pytest.mark.asyncio`, concurrency timing tests |
| `pyproject.toml` | Add `pytest-asyncio` dev dependency |

No agent files modified. No model or config changes.

## Implementation Detail: `_run_stage()`

```python
async def _run_stage(
    self, tasks: list[tuple]
) -> list:
    """Run (callable, *args) tuples concurrently. Returns results in order.
    Exceptions are returned as values (not raised) — callers must check."""
    coros = [asyncio.to_thread(fn, *args) for fn, *args in tasks]
    return await asyncio.gather(*coros, return_exceptions=True)
```

## Stage 2 — Data

```python
stage2_tasks = []
if plan.get("sql") and SkillModule.SQL_QUERYING in config.enabled_skills:
    stage2_tasks.append((sql_agent.run, request, context))
if self._spreadsheet_agent and SkillModule.SPREADSHEET_ANALYSIS in config.enabled_skills:
    stage2_tasks.append((self._spreadsheet_agent.run, request))

results = await self._run_stage(stage2_tasks)
# extract sql_data from whichever succeeded
```

## Stage 3 — Analysis

```python
stage3_tasks = []
if plan.get("chart") and sql_data:
    stage3_tasks.append(("chart", self._chart_agent.run, sql_data))
if plan.get("ml") and sql_data:
    stage3_tasks.append(("ml", self._ml_agent.run, config.client_id, request.text, sql_data))
# ... etc for anomaly, hypothesis, segment, ab_test

results = await self._run_stage([(fn, *args) for _, fn, *args in stage3_tasks])

# merge enrichment dicts
enriched_data = dict(sql_data or {})
for label, result in zip([t[0] for t in stage3_tasks], results):
    if isinstance(result, Exception):
        logger.warning("agent_failed", agent=label, error=str(result))
        continue
    if result.success:
        if label in ("hypothesis", "segment", "ab_test"):
            enriched_data.update(result.data)
        if result.chart_png:
            charts.append(result.chart_png)
```

## Stage 4 — Output

```python
stage4_tasks = []
if plan.get("report") and self._report_agent:
    stage4_tasks.append(("report", self._report_agent.run, ...))
if plan.get("deck") and self._deck_agent:
    stage4_tasks.append(("deck", self._deck_agent.run, ...))

output_results = await self._run_stage([(fn, *args) for _, fn, *args in stage4_tasks])
```

## Error Handling

- Each stage uses `return_exceptions=True` — one agent failure never aborts siblings
- Exceptions logged with agent name and error string
- Graceful degradation: response is built from whatever agents succeeded
- Behavior identical to current `if result.success:` checks — just applied after gather

## Testing

| Test | Validates |
|---|---|
| All existing orchestrator tests with `@pytest.mark.asyncio` | Backward compat |
| Stage 2 concurrency: mock sql + spreadsheet with 50ms sleep, assert elapsed < 75ms | Agents truly run in parallel |
| Stage 3 exception isolation: mock anomaly to raise, assert chart + ML results still present | `return_exceptions=True` works |
| `/analyze` endpoint smoke test | async endpoint wires correctly |

## Dependencies

- `pytest-asyncio` — new dev dependency for async test support
- No new runtime dependencies (`asyncio` and `concurrent.futures` are stdlib)
