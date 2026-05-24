# DAG Parallel Agent Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Orchestrator.process()` async and execute independent agents concurrently using asyncio stages, reducing `/analyze` latency when multiple agents are triggered.

**Architecture:** Four sequential stages; within each stage independent agents run via `asyncio.gather()` + `asyncio.to_thread()`. Agents stay synchronous — a `_run_stage()` helper accepts zero-argument callables (lambdas) and runs them concurrently. Stage 3 (analysis agents) and Stage 4 (report + deck) are the parallel stages.

**Tech Stack:** Python `asyncio` (stdlib), `asyncio.to_thread`, `asyncio.gather`, `pytest-asyncio` (already installed, `asyncio_mode = "auto"`).

---

## File Map

| File | Change |
|---|---|
| `src/orchestrator/orchestrator.py` | Add `import asyncio, logging`; add `_run_stage()` helper; `async def process()`; parallel Stage 3 + Stage 4 |
| `main.py` | `async def analyze()` + `await orchestrator.process()` |
| `src/jobs/scheduler.py` | Wrap `self._orchestrator.process()` with `asyncio.run()` |
| `tests/orchestrator/test_orchestrator.py` | All 11 `def test_` → `async def test_`; all `.process(` → `await .process(` |
| `tests/test_integration.py` | 2 tests: `async def` + `await` |
| `tests/test_integration_phase4.py` | 3 tests: `async def` + `await` |
| `tests/test_integration_phase5.py` | 5 tests: `async def` + `await` |
| `tests/test_integration_phase6.py` | 4 tests: `async def` + `await` |
| `tests/test_integration_phase7.py` | 2 tests: `async def` + `await` |
| `tests/jobs/test_scheduler.py` | Change `mock_orchestrator.process` to `AsyncMock` |

No agent files are modified.

---

### Task 1: Add `_run_stage()`, make `process()` async, update all call-site tests

**Files:**
- Modify: `src/orchestrator/orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`
- Modify: `tests/test_integration.py`
- Modify: `tests/test_integration_phase4.py`
- Modify: `tests/test_integration_phase5.py`
- Modify: `tests/test_integration_phase6.py`
- Modify: `tests/test_integration_phase7.py`

- [ ] **Step 1: Write failing test — `process()` must be awaitable**

Add this test to the bottom of `tests/orchestrator/test_orchestrator.py`:

```python
async def test_process_is_awaitable():
    deps = _make_orc_deps()
    orc = Orchestrator(**deps)
    result = await orc.process(_make_request(), _make_config(SkillModule.SQL_QUERYING))
    assert isinstance(result, Response)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py::test_process_is_awaitable -v
```

Expected: `FAILED` — `TypeError: object Response can't be used in 'await' expression` (because `process()` is currently sync).

- [ ] **Step 3: Add `import asyncio` and `_logger`, add `_run_stage()` helper, convert `process()` to `async def`**

At the top of `src/orchestrator/orchestrator.py`, add two imports after `import json`:

```python
import asyncio
import logging

_logger = logging.getLogger(__name__)
```

Add `_run_stage()` as a method on `Orchestrator` (insert before `_plan`):

```python
async def _run_stage(self, callables: list) -> list:
    return await asyncio.gather(
        *[asyncio.to_thread(fn) for fn in callables],
        return_exceptions=True,
    )
```

Change `def process(` to `async def process(` and wrap every blocking call with `await asyncio.to_thread(...)`. The complete new `process()` body — replace the entire method:

```python
async def process(
    self,
    request: Request,
    config: ClientConfig,
    clarification_state: ClarificationState | None = None,
) -> "Response | ClarificationState | AnalystResult":
    context = self._retriever.search(config.client_id, request.text)

    if (
        self._analyst_agent is not None
        and SkillModule.DEEP_ANALYSIS in config.enabled_skills
        and clarification_state is not None
        and clarification_state.deep_dive_pending
        and (
            clarification_state.deep_dive_confirmed
            or any(
                a.strip().lower() in ("yes", "y", "sure", "go ahead", "ok", "yep")
                for a in clarification_state.answers_received
            )
        )
    ):
        checkpoints: list[str] = []
        return await asyncio.to_thread(
            lambda: self._analyst_agent.run_deep(
                request=request,
                config=config,
                on_checkpoint=lambda step, thought: checkpoints.append(
                    f"Step {step}: {thought}"
                ),
            )
        )

    state = await asyncio.to_thread(
        self._clarifier.check, request, context, clarification_state
    )

    if not state.is_resolved:
        return state

    if (
        self._analyst_agent is not None
        and SkillModule.DEEP_ANALYSIS in config.enabled_skills
        and clarification_state is None
        and self._detect_deep_intent(request)
    ):
        deep_state = ClarificationState(original_request=request)
        deep_state.questions_asked = [
            "Want me to do a deep dive on this? It may take a few minutes. "
            "Reply yes to proceed or no for a quick answer."
        ]
        deep_state.deep_dive_pending = True
        deep_state.is_resolved = False
        return deep_state

    # Stage 1 — Plan
    plan = await asyncio.to_thread(self._plan, request, context)
    charts: list[bytes] = []
    sql_data: dict | None = None
    sql_queries: list[str] = []
    anomalies: list[dict] = []

    # Spreadsheet — alternative data source when file attached (sequential)
    if (
        request.file_bytes is not None
        and request.filename is not None
        and self._spreadsheet_agent is not None
        and SkillModule.SPREADSHEET_ANALYSIS in config.enabled_skills
    ):
        sheet_result = await asyncio.to_thread(
            lambda: self._spreadsheet_agent.run(
                file_bytes=request.file_bytes,
                filename=request.filename,
                question=request.text,
                config=config,
            )
        )
        if sheet_result.success:
            sql_data = sheet_result.data

    # Funnel KB lookup (sequential — may return early with clarification)
    if plan.get("funnel") and SkillModule.FUNNEL_ANALYSIS in config.enabled_skills:
        funnel_docs = self._retriever.search(
            config.client_id, "funnel stages conversion steps flow"
        )
        if funnel_docs:
            context = context + funnel_docs[:2]
        else:
            funnel_state = ClarificationState(original_request=request)
            funnel_state.questions_asked = [
                "I don't have your funnel stages defined. What are the conversion steps? "
                "(e.g., Visit → Signup → Active User → Paid Customer)"
            ]
            funnel_state.is_resolved = False
            return funnel_state

    # SQL — sequential data fetch
    if plan.get("sql") and SkillModule.SQL_QUERYING in config.enabled_skills:
        analysis_mode = None
        if plan.get("funnel") and SkillModule.FUNNEL_ANALYSIS in config.enabled_skills:
            analysis_mode = "funnel"
        elif plan.get("cohort") and SkillModule.COHORT_ANALYSIS in config.enabled_skills:
            analysis_mode = "cohort"

        sql_agent = self._sql_agent
        if self._registry is not None:
            connector = self._registry.get_connector(config.client_id)
            if connector is not None:
                sql_agent = SQLAgent(llm=self._llm, connector=connector)

        if sql_agent is not None:
            _mode = analysis_mode
            sql_result = await asyncio.to_thread(
                lambda: sql_agent.run(
                    config.client_id, request.text, context, analysis_mode=_mode
                )
            )
            if sql_result.success:
                sql_data = sql_result.data
                if sql_data and sql_data.get("query"):
                    sql_queries.append(sql_data["query"])

    # Stage 3 — Analysis agents (sequential for now; parallelised in Task 2)
    if sql_data:
        anomaly_skill = (
            SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
            or SkillModule.STATISTICAL_ANOMALY in config.enabled_skills
        )
        if self._chart_agent is not None and SkillModule.DATA_VISUALIZATION in config.enabled_skills:
            chart_result = await asyncio.to_thread(
                lambda: self._chart_agent.run(sql_data)
            )
            if chart_result.success and chart_result.chart_png:
                charts.append(chart_result.chart_png)

        if self._anomaly_agent is not None and anomaly_skill:
            mode = (
                "both"
                if SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
                and SkillModule.STATISTICAL_ANOMALY in config.enabled_skills
                else "hard"
                if SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
                else "statistical"
            )
            _amode = mode
            anomaly_result = await asyncio.to_thread(
                lambda: self._anomaly_agent.run(config.client_id, sql_data, mode=_amode)
            )
            if anomaly_result.success:
                anomalies = anomaly_result.data.get("anomalies", [])

        if (
            plan.get("ml")
            and self._ml_agent is not None
            and SkillModule.MACHINE_LEARNING in config.enabled_skills
        ):
            ml_result = await asyncio.to_thread(
                lambda: self._ml_agent.run(config.client_id, request.text, sql_data)
            )
            if ml_result.success and ml_result.chart_png:
                charts.append(ml_result.chart_png)

        if (
            plan.get("hypothesis")
            and self._hypothesis_agent is not None
            and SkillModule.HYPOTHESIS_TESTING in config.enabled_skills
        ):
            hyp_result = await asyncio.to_thread(
                lambda: self._hypothesis_agent.run(config.client_id, request.text, sql_data)
            )
            if hyp_result.success:
                sql_data = {**(sql_data or {}), **hyp_result.data}

        if (
            plan.get("segment")
            and self._segmentation_agent is not None
            and SkillModule.SEGMENTATION in config.enabled_skills
        ):
            seg_result = await asyncio.to_thread(
                lambda: self._segmentation_agent.run(config.client_id, request.text, sql_data)
            )
            if seg_result.success:
                if seg_result.chart_png:
                    charts.append(seg_result.chart_png)
                sql_data = {**(sql_data or {}), **seg_result.data}

        if (
            plan.get("ab_test")
            and self._ab_agent is not None
            and SkillModule.AB_TESTING in config.enabled_skills
        ):
            ab_result = await asyncio.to_thread(
                lambda: self._ab_agent.run(config.client_id, request.text, sql_data)
            )
            if ab_result.success:
                sql_data = {**(sql_data or {}), **ab_result.data}

    # Stage 3b — Generate response (sequential LLM call)
    text = await asyncio.to_thread(
        self._generate_response, request, context, sql_data, state.assumptions
    )

    # Stage 4 — Output agents (sequential for now; parallelised in Task 3)
    report_markdown: str | None = None
    report_html: str | None = None
    deck_pptx: bytes | None = None

    if (
        plan.get("report")
        and self._report_agent is not None
        and SkillModule.REPORT_GENERATION in config.enabled_skills
    ):
        report_result = await asyncio.to_thread(
            lambda: self._report_agent.run(
                config.client_id,
                request.text,
                {
                    "analysis_text": text,
                    "sql_rows": (sql_data.get("rows", []) or [])[:10] if sql_data else [],
                    "anomalies": anomalies,
                },
            )
        )
        if report_result.success:
            report_markdown = report_result.data["markdown"]
            report_html = report_result.data["html"]

    if (
        plan.get("deck")
        and self._deck_agent is not None
        and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
    ):
        findings = (
            [f"{r}" for r in (sql_data.get("rows", []) or [])[:5]] if sql_data else []
        )
        storyline = await asyncio.to_thread(
            lambda: self._deck_agent.build_storyline(
                analysis_text=text,
                findings=findings,
                solutions=[],
                recommendation="",
            )
        )
        deck_result = await asyncio.to_thread(
            lambda: self._deck_agent.run(
                title=request.text[:100], storyline=storyline, charts=charts
            )
        )
        if deck_result.success:
            deck_pptx = deck_result.deck_pptx

    response = Response(
        request_id=str(uuid.uuid4()),
        text=text,
        charts=charts,
        assumptions=state.assumptions,
        deck_pptx=deck_pptx,
        anomalies=[Anomaly(**a) for a in anomalies],
        report_markdown=report_markdown,
        report_html=report_html,
    )

    if self._memory_logger is not None:
        self._memory_logger.log(
            request=request,
            response=response,
            sql_queries=sql_queries,
        )

    return response
```

- [ ] **Step 4: Run the new test to confirm it passes**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py::test_process_is_awaitable -v
```

Expected: `PASSED`.

- [ ] **Step 5: Update all existing orchestrator tests to be async**

In `tests/orchestrator/test_orchestrator.py`, convert every `def test_` to `async def test_` and every direct `.process(` call to `await .process(`. There are 11 tests and ~11 `.process(` calls.

Use your editor's find-and-replace within this file only:
- `def test_` → `async def test_`
- `= orc.process(` → `= await orc.process(`
- `= orch.process(` → `= await orch.process(`
- `orc.process(` (no assignment) → `await orc.process(`
- `orchestrator.process(` → `await orchestrator.process(`

- [ ] **Step 6: Update integration test files to be async**

Apply the same find-and-replace to these files (only for tests that call a real orchestrator — not mock orchestrators):

`tests/test_integration.py` — 2 tests call `orchestrator.process(`:
- `def test_` → `async def test_`
- `orchestrator.process(` → `await orchestrator.process(`

`tests/test_integration_phase4.py` — 3 tests call `orc.process(`:
- `def test_` → `async def test_`
- `orc.process(` → `await orc.process(`

`tests/test_integration_phase5.py` — 5 tests call `orc.process(`:
- `def test_` → `async def test_`
- `orc.process(` → `await orc.process(`

`tests/test_integration_phase6.py` — 4 tests call `orch.process(`:
- `def test_` → `async def test_`
- `orch.process(` → `await orch.process(`

`tests/test_integration_phase7.py` — 2 tests call `orch.process(`:
- `def test_` → `async def test_`
- `orch.process(` → `await orch.process(`

Do NOT modify `test_integration_phase2.py` or `test_integration_phase3.py` — they use a `mock_orchestrator` and do not await anything.

- [ ] **Step 7: Run the full test suite**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all tests pass (same count as before Task 1, plus the new `test_process_is_awaitable`).

- [ ] **Step 8: Commit**

```bash
git add src/orchestrator/orchestrator.py \
        tests/orchestrator/test_orchestrator.py \
        tests/test_integration.py \
        tests/test_integration_phase4.py \
        tests/test_integration_phase5.py \
        tests/test_integration_phase6.py \
        tests/test_integration_phase7.py
git commit -m "refactor: convert orchestrator.process() to async with asyncio.to_thread wrappers"
```

---

### Task 2: Parallelise Stage 3 — analysis agents with exception isolation

**Files:**
- Modify: `src/orchestrator/orchestrator.py` (Stage 3 section only)
- Modify: `tests/orchestrator/test_orchestrator.py` (add 1 new test)

- [ ] **Step 1: Write the failing isolation test**

Add to the bottom of `tests/orchestrator/test_orchestrator.py`:

```python
async def test_stage3_exception_does_not_abort_other_agents():
    """If one Stage 3 agent raises, others still run and their results appear."""
    deps = _make_orc_deps()

    # anomaly raises
    mock_anomaly = MagicMock()
    mock_anomaly.run.side_effect = RuntimeError("anomaly crashed")

    # chart returns a png
    mock_chart = MagicMock()
    mock_chart.run.return_value = AgentResult(
        agent_name="chart_agent", success=True, chart_png=b"fakepng"
    )

    orc = Orchestrator(
        **{k: v for k, v in deps.items() if k not in ("chart_agent",)},
        chart_agent=mock_chart,
        anomaly_agent=mock_anomaly,
    )
    result = await orc.process(
        _make_request(),
        _make_config(
            SkillModule.SQL_QUERYING,
            SkillModule.DATA_VISUALIZATION,
            SkillModule.HARD_RULE_ANOMALY,
        ),
    )
    assert result.charts, "chart result should be present despite anomaly failure"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py::test_stage3_exception_does_not_abort_other_agents -v
```

Expected: `FAILED` — the `RuntimeError` propagates from the sequential `await asyncio.to_thread(lambda: self._anomaly_agent.run(...))` and aborts the function before `chart_agent` runs.

- [ ] **Step 3: Replace sequential Stage 3 with parallel `_run_stage()`**

In `src/orchestrator/orchestrator.py`, find the block starting with `# Stage 3 — Analysis agents (sequential for now` and replace the entire `if sql_data:` block with:

```python
    # Stage 3 — Analysis agents (parallel)
    if sql_data:
        stage3_fns: list = []
        stage3_labels: list[str] = []

        if self._chart_agent is not None and SkillModule.DATA_VISUALIZATION in config.enabled_skills:
            _chart = self._chart_agent
            _sd = sql_data
            stage3_fns.append(lambda: _chart.run(_sd))
            stage3_labels.append("chart")

        anomaly_skill = (
            SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
            or SkillModule.STATISTICAL_ANOMALY in config.enabled_skills
        )
        if self._anomaly_agent is not None and anomaly_skill:
            _amode = (
                "both"
                if SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
                and SkillModule.STATISTICAL_ANOMALY in config.enabled_skills
                else "hard"
                if SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
                else "statistical"
            )
            _anomaly = self._anomaly_agent
            _cid = config.client_id
            _sd2 = sql_data
            stage3_fns.append(lambda: _anomaly.run(_cid, _sd2, mode=_amode))
            stage3_labels.append("anomaly")

        if (
            plan.get("ml")
            and self._ml_agent is not None
            and SkillModule.MACHINE_LEARNING in config.enabled_skills
        ):
            _ml = self._ml_agent
            _cid2 = config.client_id
            _txt = request.text
            _sd3 = sql_data
            stage3_fns.append(lambda: _ml.run(_cid2, _txt, _sd3))
            stage3_labels.append("ml")

        if (
            plan.get("hypothesis")
            and self._hypothesis_agent is not None
            and SkillModule.HYPOTHESIS_TESTING in config.enabled_skills
        ):
            _hyp = self._hypothesis_agent
            _cid3 = config.client_id
            _txt2 = request.text
            _sd4 = sql_data
            stage3_fns.append(lambda: _hyp.run(_cid3, _txt2, _sd4))
            stage3_labels.append("hypothesis")

        if (
            plan.get("segment")
            and self._segmentation_agent is not None
            and SkillModule.SEGMENTATION in config.enabled_skills
        ):
            _seg = self._segmentation_agent
            _cid4 = config.client_id
            _txt3 = request.text
            _sd5 = sql_data
            stage3_fns.append(lambda: _seg.run(_cid4, _txt3, _sd5))
            stage3_labels.append("segment")

        if (
            plan.get("ab_test")
            and self._ab_agent is not None
            and SkillModule.AB_TESTING in config.enabled_skills
        ):
            _ab = self._ab_agent
            _cid5 = config.client_id
            _txt4 = request.text
            _sd6 = sql_data
            stage3_fns.append(lambda: _ab.run(_cid5, _txt4, _sd6))
            stage3_labels.append("ab_test")

        stage3_results = await self._run_stage(stage3_fns)

        enriched: dict = dict(sql_data)
        for label, result in zip(stage3_labels, stage3_results):
            if isinstance(result, Exception):
                _logger.warning("agent_failed agent=%s error=%s", label, result)
                continue
            if not result.success:
                continue
            if label == "chart":
                if result.chart_png:
                    charts.append(result.chart_png)
            elif label == "anomaly":
                anomalies = result.data.get("anomalies", [])
            elif label == "ml":
                if result.chart_png:
                    charts.append(result.chart_png)
            elif label in ("hypothesis", "segment", "ab_test"):
                enriched.update(result.data)
                if result.chart_png:
                    charts.append(result.chart_png)

        sql_data = enriched
```

- [ ] **Step 4: Run the isolation test**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py::test_stage3_exception_does_not_abort_other_agents -v
```

Expected: `PASSED`.

- [ ] **Step 5: Run the full test suite**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all previous tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/orchestrator.py tests/orchestrator/test_orchestrator.py
git commit -m "feat: parallelise Stage 3 analysis agents with asyncio.gather"
```

---

### Task 3: Parallelise Stage 4 — report + deck output agents

**Files:**
- Modify: `src/orchestrator/orchestrator.py` (Stage 4 section only)
- Modify: `tests/orchestrator/test_orchestrator.py` (add 1 new test)

- [ ] **Step 1: Write the failing timing test**

Add to the bottom of `tests/orchestrator/test_orchestrator.py`:

```python
import time as _time

async def test_stage4_output_agents_run_in_parallel():
    """Report and deck agents run concurrently — total time < sequential sum."""
    from unittest.mock import MagicMock

    deps = _make_orc_deps()
    # plan returns report=True, deck=True
    deps["llm"].complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": true, '
        '"funnel": false, "cohort": false, "hypothesis": false, '
        '"segment": false, "ab_test": false, "report": true}',
        "Analysis complete.",
    ]

    def _slow_report(*args, **kwargs):
        _time.sleep(0.05)
        return AgentResult(
            agent_name="report_agent", success=True,
            data={"markdown": "# R", "html": "<h1>R</h1>"},
        )

    def _slow_deck(*args, **kwargs):
        _time.sleep(0.05)
        return AgentResult(agent_name="deck_agent", success=True, deck_pptx=b"pptx")

    mock_report = MagicMock()
    mock_report.run.side_effect = _slow_report

    mock_deck = MagicMock()
    mock_deck.run.side_effect = _slow_deck
    mock_deck.build_storyline.return_value = MagicMock()

    orc = Orchestrator(
        **deps,
        report_agent=mock_report,
        deck_agent=mock_deck,
    )
    config = _make_config(
        SkillModule.SQL_QUERYING,
        SkillModule.REPORT_GENERATION,
        SkillModule.PRESENTATION_BUILDING,
    )

    start = _time.monotonic()
    result = await orc.process(_make_request(text="generate a report and a deck"), config)
    elapsed = _time.monotonic() - start

    assert elapsed < 0.09, f"Expected parallel (~50ms), got {elapsed:.3f}s (sequential would be ~100ms)"
    assert result.report_markdown is not None
    assert result.deck_pptx is not None
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py::test_stage4_output_agents_run_in_parallel -v
```

Expected: `FAILED` — `elapsed` is ~0.10s (sequential) not < 0.09s.

- [ ] **Step 3: Replace sequential Stage 4 with parallel `_run_stage()`**

In `src/orchestrator/orchestrator.py`, find the block starting with `# Stage 4 — Output agents (sequential for now` and replace the entire section (from after `text = await ...` down to `if self._memory_logger`) with:

```python
    # Stage 4 — Output agents (parallel)
    report_markdown: str | None = None
    report_html: str | None = None
    deck_pptx: bytes | None = None

    stage4_fns: list = []
    stage4_labels: list[str] = []

    if (
        plan.get("report")
        and self._report_agent is not None
        and SkillModule.REPORT_GENERATION in config.enabled_skills
    ):
        _report = self._report_agent
        _cid_r = config.client_id
        _txt_r = request.text
        _sd_r = sql_data
        _anom_r = anomalies
        _text_r = text
        stage4_fns.append(
            lambda: _report.run(
                _cid_r,
                _txt_r,
                {
                    "analysis_text": _text_r,
                    "sql_rows": (_sd_r.get("rows", []) or [])[:10] if _sd_r else [],
                    "anomalies": _anom_r,
                },
            )
        )
        stage4_labels.append("report")

    if (
        plan.get("deck")
        and self._deck_agent is not None
        and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
    ):
        findings = (
            [f"{r}" for r in (sql_data.get("rows", []) or [])[:5]] if sql_data else []
        )
        storyline = await asyncio.to_thread(
            lambda: self._deck_agent.build_storyline(
                analysis_text=text,
                findings=findings,
                solutions=[],
                recommendation="",
            )
        )
        _deck = self._deck_agent
        _title = request.text[:100]
        _story = storyline
        _charts = charts
        stage4_fns.append(
            lambda: _deck.run(title=_title, storyline=_story, charts=_charts)
        )
        stage4_labels.append("deck")

    if stage4_fns:
        stage4_results = await self._run_stage(stage4_fns)
        for label, result in zip(stage4_labels, stage4_results):
            if isinstance(result, Exception):
                _logger.warning("agent_failed agent=%s error=%s", label, result)
                continue
            if not result.success:
                continue
            if label == "report":
                report_markdown = result.data["markdown"]
                report_html = result.data["html"]
            elif label == "deck":
                deck_pptx = result.deck_pptx
```

- [ ] **Step 4: Run the timing test**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py::test_stage4_output_agents_run_in_parallel -v
```

Expected: `PASSED` — elapsed ~50ms.

- [ ] **Step 5: Run the full test suite**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/orchestrator.py tests/orchestrator/test_orchestrator.py
git commit -m "feat: parallelise Stage 4 output agents (report + deck) with asyncio.gather"
```

---

### Task 4: Update `main.py` and `scheduler.py` call sites

**Files:**
- Modify: `main.py`
- Modify: `src/jobs/scheduler.py`
- Modify: `tests/jobs/test_scheduler.py`

- [ ] **Step 1: Make `/analyze` endpoint async in `main.py`**

Find this function in `main.py` (around line 212):

```python
def analyze(request: StarletteRequest, body: AnalyzeBody, x_client_id: str = Header(default="")):
    if body.request.client_id != x_client_id:
        raise HTTPException(status_code=403, detail="client_id mismatch")
    config = registry.get(body.request.client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    result = orchestrator.process(body.request, config, body.clarification_state)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"result": str(result)}
```

Replace with:

```python
async def analyze(request: StarletteRequest, body: AnalyzeBody, x_client_id: str = Header(default="")):
    if body.request.client_id != x_client_id:
        raise HTTPException(status_code=403, detail="client_id mismatch")
    config = registry.get(body.request.client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    result = await orchestrator.process(body.request, config, body.clarification_state)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"result": str(result)}
```

- [ ] **Step 2: Update `scheduler.py` to run the coroutine**

Find this line in `src/jobs/scheduler.py` (around line 56):

```python
result = self._orchestrator.process(request, config)
```

Replace with:

```python
import asyncio
result = asyncio.run(self._orchestrator.process(request, config))
```

Move the `import asyncio` to the top of the file with the other imports (not inline).

- [ ] **Step 3: Update the scheduler test to use `AsyncMock`**

In `tests/jobs/test_scheduler.py`, find the mock orchestrator setup. It currently looks like:

```python
mock_orchestrator = MagicMock()
mock_orchestrator.process.return_value = Response(...)
```

Change it to use `AsyncMock`:

```python
from unittest.mock import AsyncMock, MagicMock
mock_orchestrator = MagicMock()
mock_orchestrator.process = AsyncMock(return_value=Response(...))
```

- [ ] **Step 4: Run `/analyze` HTTP tests**

```bash
.venv/bin/python -m pytest tests/test_api.py tests/test_api_phase11.py -v
```

Expected: all pass. The `TestClient` from FastAPI handles both sync and async routes.

- [ ] **Step 5: Run the scheduler tests**

```bash
.venv/bin/python -m pytest tests/jobs/ -v
```

Expected: all pass.

- [ ] **Step 6: Run the full test suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass (count should match or exceed prior count).

- [ ] **Step 7: Commit**

```bash
git add main.py src/jobs/scheduler.py tests/jobs/test_scheduler.py
git commit -m "feat: wire async orchestrator into /analyze endpoint and scheduler"
```
