# Phase 4: Anomaly Agent + Interaction Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add anomaly detection (hard-rule + statistical) and interaction memory logging to the AI analyst agent.

**Architecture:** `AnomalyAgent` evaluates hard rules (stored as JSON in ChromaDB) and computes z-score outliers on SQL result data. `InteractionMemoryLogger` appends every request+response as a document into the client's ChromaDB collection so future RAG searches benefit from prior interactions. Both are wired into `Orchestrator.process()` as post-SQL steps. `SQLAgent` gets a new `analysis_mode` parameter to produce funnel and cohort SQL patterns.

**Tech Stack:** Python 3.11, numpy (already installed), ChromaDB (already installed), pydantic v2. No new dependencies.

---

## File Structure

**New files:**
- `src/agents/anomaly_agent.py` — hard-rule + statistical anomaly detection
- `src/knowledge/interaction_memory.py` — ChromaDB interaction logger
- `tests/agents/test_anomaly_agent.py`
- `tests/knowledge/test_interaction_memory.py`
- `tests/test_integration_phase4.py`

**Modified files:**
- `src/core/models.py` — add `Anomaly` model, `anomalies` on `Response`, `FUNNEL_ANALYSIS` + `COHORT_ANALYSIS` in `SkillModule`
- `src/agents/sql_agent.py` — add `analysis_mode` param for funnel/cohort prompt hint
- `src/orchestrator/orchestrator.py` — wire `AnomalyAgent`, `InteractionMemoryLogger`, funnel/cohort plan keys
- `main.py` — instantiate and wire new components

---

### Task 1: Model extensions — Anomaly + SkillModule entries

**Files:**
- Modify: `src/core/models.py`
- Test: `tests/core/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# add to tests/core/test_models.py

from src.core.models import Anomaly, Response, SkillModule

def test_anomaly_model_fields():
    a = Anomaly(
        metric="churn_rate",
        value=0.15,
        threshold=0.10,
        operator=">",
        severity="critical",
        description="churn_rate is 0.15 (rule: churn_rate > 0.1)",
        mode="hard_rule",
    )
    assert a.metric == "churn_rate"
    assert a.severity == "critical"
    assert a.mode == "hard_rule"

def test_anomaly_defaults():
    a = Anomaly(metric="revenue", value=500.0, description="outlier", mode="statistical")
    assert a.severity == "warning"
    assert a.expected is None
    assert a.threshold is None
    assert a.operator is None

def test_response_anomalies_defaults_empty():
    r = Response(request_id="r1", text="ok")
    assert r.anomalies == []

def test_response_accepts_anomalies():
    a = Anomaly(metric="x", value=1.0, description="test", mode="hard_rule")
    r = Response(request_id="r1", text="ok", anomalies=[a])
    assert len(r.anomalies) == 1
    assert r.anomalies[0].metric == "x"

def test_skill_module_has_funnel_and_cohort():
    assert SkillModule.FUNNEL_ANALYSIS == "funnel_analysis"
    assert SkillModule.COHORT_ANALYSIS == "cohort_analysis"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_models.py::test_anomaly_model_fields tests/core/test_models.py::test_response_anomalies_defaults_empty tests/core/test_models.py::test_skill_module_has_funnel_and_cohort -v
```

Expected: FAIL with `ImportError: cannot import name 'Anomaly'`

- [ ] **Step 3: Add Anomaly model and update SkillModule + Response in models.py**

In `src/core/models.py`, add after `class TicketRef`:

```python
class Anomaly(BaseModel):
    metric: str
    value: float
    expected: Optional[float] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    severity: str = "warning"
    description: str
    mode: str  # "hard_rule" or "statistical"
```

Add to `SkillModule` enum:

```python
    FUNNEL_ANALYSIS = "funnel_analysis"
    COHORT_ANALYSIS = "cohort_analysis"
```

Update `Response` — change `anomalies` field:

```python
class Response(BaseModel):
    request_id: str
    text: str
    charts: list[bytes] = []
    assumptions: list[str] = []
    ticket: Optional[TicketRef] = None
    deck_pptx: Optional[bytes] = None
    anomalies: list["Anomaly"] = []
```

Add `from __future__ import annotations` at top of file if not present, or use the string forward reference as shown.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/core/test_models.py -v
```

Expected: all tests PASS (including the 5 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/core/models.py tests/core/test_models.py
git commit -m "feat: add Anomaly model, anomalies on Response, FUNNEL/COHORT SkillModules"
```

---

### Task 2: AnomalyAgent — hard-rule and statistical detection

**Files:**
- Create: `src/agents/anomaly_agent.py`
- Create: `tests/agents/test_anomaly_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_anomaly_agent.py
import json
import pytest
from unittest.mock import MagicMock

from src.agents.anomaly_agent import AnomalyAgent
from src.core.models import AgentResult


def _agent_with_rules(*rule_jsons):
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [json.dumps(r) for r in rule_jsons]
    return AnomalyAgent(retriever=mock_retriever)


def _agent_no_rules():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    return AnomalyAgent(retriever=mock_retriever)


def test_hard_rule_detects_breach():
    agent = _agent_with_rules(
        {"metric": "churn_rate", "operator": ">", "threshold": 0.1, "severity": "critical"}
    )
    result = agent.run(
        client_id="c1",
        data={"columns": ["churn_rate"], "rows": [{"churn_rate": 0.15}]},
        mode="hard",
    )
    assert result.success is True
    anomalies = result.data["anomalies"]
    assert len(anomalies) == 1
    assert anomalies[0]["metric"] == "churn_rate"
    assert anomalies[0]["severity"] == "critical"
    assert anomalies[0]["mode"] == "hard_rule"


def test_hard_rule_no_breach():
    agent = _agent_with_rules(
        {"metric": "churn_rate", "operator": ">", "threshold": 0.2, "severity": "warning"}
    )
    result = agent.run(
        client_id="c1",
        data={"columns": ["churn_rate"], "rows": [{"churn_rate": 0.15}]},
        mode="hard",
    )
    assert result.success is True
    assert result.data["anomalies"] == []


def test_hard_rule_all_operators():
    for op, val, threshold, should_flag in [
        (">", 5, 4, True), (">", 4, 5, False),
        (">=", 5, 5, True), (">=", 4, 5, False),
        ("<", 3, 4, True), ("<", 5, 4, False),
        ("<=", 4, 4, True), ("<=", 5, 4, False),
    ]:
        agent = _agent_with_rules({"metric": "m", "operator": op, "threshold": threshold})
        result = agent.run("c1", {"columns": ["m"], "rows": [{"m": val}]}, mode="hard")
        assert (len(result.data["anomalies"]) > 0) == should_flag, f"op={op} val={val} threshold={threshold}"


def test_invalid_rule_json_skipped():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        "not valid json",
        json.dumps({"metric": "revenue", "operator": ">", "threshold": 1000}),
    ]
    agent = AnomalyAgent(retriever=mock_retriever)
    result = agent.run(
        client_id="c1",
        data={"columns": ["revenue"], "rows": [{"revenue": 1500}]},
        mode="hard",
    )
    assert result.success is True
    assert len(result.data["anomalies"]) == 1


def test_statistical_detects_outlier():
    agent = _agent_no_rules()
    data = {
        "columns": ["revenue"],
        "rows": [
            {"revenue": 100}, {"revenue": 102}, {"revenue": 98},
            {"revenue": 99}, {"revenue": 101}, {"revenue": 500},
        ],
    }
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.success is True
    anomalies = result.data["anomalies"]
    assert any(a["value"] == 500 and a["metric"] == "revenue" for a in anomalies)


def test_statistical_no_anomaly_in_uniform_data():
    agent = _agent_no_rules()
    data = {
        "columns": ["revenue"],
        "rows": [{"revenue": v} for v in [100, 102, 98, 99, 101, 103]],
    }
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.data["anomalies"] == []


def test_statistical_skips_non_numeric_columns():
    agent = _agent_no_rules()
    data = {
        "columns": ["name", "revenue"],
        "rows": [
            {"name": "Alice", "revenue": 100},
            {"name": "Bob", "revenue": 102},
            {"name": "Charlie", "revenue": 99},
        ],
    }
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.success is True
    assert all(a["metric"] != "name" for a in result.data["anomalies"])


def test_both_mode_runs_both():
    agent = _agent_with_rules(
        {"metric": "churn_rate", "operator": ">", "threshold": 0.05, "severity": "warning"}
    )
    data = {
        "columns": ["churn_rate", "revenue"],
        "rows": [
            {"churn_rate": 0.10, "revenue": 100},
            {"churn_rate": 0.09, "revenue": 102},
            {"churn_rate": 0.11, "revenue": 600},
        ],
    }
    result = agent.run(client_id="c1", data=data, mode="both")
    modes = {a["mode"] for a in result.data["anomalies"]}
    assert "hard_rule" in modes
    assert "statistical" in modes


def test_insufficient_data_for_statistical_returns_no_anomalies():
    agent = _agent_no_rules()
    data = {"columns": ["revenue"], "rows": [{"revenue": 100}, {"revenue": 200}]}
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.success is True
    assert result.data["anomalies"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_anomaly_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.anomaly_agent'`

- [ ] **Step 3: Implement AnomalyAgent**

```python
# src/agents/anomaly_agent.py
import json
import numpy as np

from src.core.models import AgentResult, Anomaly
from src.knowledge.retriever import KnowledgeRetriever


class AnomalyAgent:
    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    def run(
        self,
        client_id: str,
        data: dict,
        mode: str = "both",
    ) -> AgentResult:
        anomalies: list[Anomaly] = []
        if mode in ("hard", "both"):
            anomalies.extend(self._check_hard_rules(client_id, data))
        if mode in ("statistical", "both"):
            anomalies.extend(self._check_statistical(data))
        return AgentResult(
            agent_name="anomaly_agent",
            success=True,
            data={"anomalies": [a.model_dump() for a in anomalies]},
        )

    def _check_hard_rules(self, client_id: str, data: dict) -> list[Anomaly]:
        docs = self._retriever.search(client_id, "anomaly rules thresholds metrics")
        anomalies: list[Anomaly] = []
        for doc in docs:
            try:
                rule = json.loads(doc)
                if not all(k in rule for k in ("metric", "operator", "threshold")):
                    continue
                metric = rule["metric"]
                op = rule["operator"]
                threshold = float(rule["threshold"])
                severity = rule.get("severity", "warning")
                for row in data.get("rows", []):
                    if metric not in row:
                        continue
                    try:
                        value = float(row[metric])
                    except (TypeError, ValueError):
                        continue
                    breached = (
                        (op == ">" and value > threshold)
                        or (op == ">=" and value >= threshold)
                        or (op == "<" and value < threshold)
                        or (op == "<=" and value <= threshold)
                    )
                    if breached:
                        anomalies.append(Anomaly(
                            metric=metric,
                            value=value,
                            threshold=threshold,
                            operator=op,
                            severity=severity,
                            description=f"{metric} is {value} (rule: {metric} {op} {threshold})",
                            mode="hard_rule",
                        ))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        return anomalies

    def _check_statistical(self, data: dict) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        rows = data.get("rows", [])
        if len(rows) < 3:
            return anomalies
        for col in data.get("columns", []):
            values = [
                float(row[col])
                for row in rows
                if col in row and isinstance(row[col], (int, float))
            ]
            if len(values) < 3:
                continue
            arr = np.array(values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            if std == 0:
                continue
            for val in values:
                z = abs((val - mean) / std)
                if z > 2.5:
                    anomalies.append(Anomaly(
                        metric=col,
                        value=val,
                        expected=mean,
                        severity="critical" if z >= 3.0 else "warning",
                        description=(
                            f"{col} value {val:.2f} is {z:.1f} std devs "
                            f"from mean ({mean:.2f})"
                        ),
                        mode="statistical",
                    ))
        return anomalies
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_anomaly_agent.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/anomaly_agent.py tests/agents/test_anomaly_agent.py
git commit -m "feat: add AnomalyAgent with hard-rule and statistical detection"
```

---

### Task 3: InteractionMemoryLogger

**Files:**
- Create: `src/knowledge/interaction_memory.py`
- Create: `tests/knowledge/test_interaction_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/knowledge/test_interaction_memory.py
import json
import pytest
from unittest.mock import MagicMock, call

from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.core.models import Channel, Request, Response


def _make_request(text="show churn", client_id="c1"):
    return Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Ana",
        text=text,
        timestamp="2026-05-14T00:00:00",
        client_id=client_id,
    )


def test_log_calls_store_add():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="Churn is 5%.")
    logger.log(req, resp)
    mock_store.add.assert_called_once()


def test_log_uses_correct_client_id():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request(client_id="client-xyz")
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp)
    args = mock_store.add.call_args[0]
    assert args[0] == "client-xyz"


def test_log_document_contains_request_and_response():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request(text="show revenue")
    resp = Response(request_id="r1", text="Revenue is $1M.")
    logger.log(req, resp)
    doc_str = mock_store.add.call_args[0][1][0]
    doc = json.loads(doc_str)
    assert doc["request_text"] == "show revenue"
    assert doc["response_text"] == "Revenue is $1M."
    assert doc["channel"] == "slack"


def test_log_includes_sql_queries():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp, sql_queries=["SELECT churn FROM metrics"])
    doc = json.loads(mock_store.add.call_args[0][1][0])
    assert doc["sql_queries"] == ["SELECT churn FROM metrics"]


def test_log_metadata_has_type_interaction_memory():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp)
    metadatas = mock_store.add.call_args[0][3]
    assert metadatas[0]["type"] == "interaction_memory"


def test_log_generates_unique_ids():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp)
    logger.log(req, resp)
    ids_first = mock_store.add.call_args_list[0][0][2]
    ids_second = mock_store.add.call_args_list[1][0][2]
    assert ids_first[0] != ids_second[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/knowledge/test_interaction_memory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge.interaction_memory'`

- [ ] **Step 3: Implement InteractionMemoryLogger**

```python
# src/knowledge/interaction_memory.py
import json
import uuid

from src.core.models import Request, Response
from src.knowledge.vector_store import VectorStore


class InteractionMemoryLogger:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def log(
        self,
        request: Request,
        response: Response,
        sql_queries: list[str] | None = None,
        corrections: list[str] | None = None,
    ) -> None:
        doc = json.dumps({
            "request_text": request.text,
            "response_text": response.text,
            "sql_queries": sql_queries or [],
            "corrections": corrections or [],
            "timestamp": request.timestamp,
            "channel": request.channel.value,
            "client_id": request.client_id,
        })
        self._store.add(
            request.client_id,
            [doc],
            [str(uuid.uuid4())],
            [{"type": "interaction_memory", "client_id": request.client_id}],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/knowledge/test_interaction_memory.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/interaction_memory.py tests/knowledge/test_interaction_memory.py
git commit -m "feat: add InteractionMemoryLogger for ChromaDB interaction history"
```

---

### Task 4: SQLAgent funnel/cohort analysis_mode

**Files:**
- Modify: `src/agents/sql_agent.py`
- Modify: `tests/agents/test_sql_agent.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/agents/test_sql_agent.py`:

```python
def test_funnel_mode_adds_hint_to_system_prompt():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT stage, COUNT(*) FROM funnel GROUP BY stage"
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {}
    mock_connector.execute.return_value = __import__('pandas').DataFrame(
        {"stage": ["A", "B"], "count": [100, 80]}
    )
    mock_store = MagicMock()
    from src.agents.sql_agent import SQLAgent
    agent = SQLAgent(llm=mock_llm, connector=mock_connector, store=mock_store)
    agent.run("c1", "show funnel", [], analysis_mode="funnel")
    system_prompt = mock_llm.complete.call_args[0][1]
    assert "funnel" in system_prompt.lower()
    assert "conversion" in system_prompt.lower()


def test_cohort_mode_adds_hint_to_system_prompt():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT cohort, period, retention FROM cohorts"
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {}
    mock_connector.execute.return_value = __import__('pandas').DataFrame(
        {"cohort": ["Jan"], "period": [1], "retention": [0.8]}
    )
    mock_store = MagicMock()
    from src.agents.sql_agent import SQLAgent
    agent = SQLAgent(llm=mock_llm, connector=mock_connector, store=mock_store)
    agent.run("c1", "show cohort retention", [], analysis_mode="cohort")
    system_prompt = mock_llm.complete.call_args[0][1]
    assert "cohort" in system_prompt.lower()
    assert "retention" in system_prompt.lower()


def test_no_analysis_mode_unchanged():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT 1"
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {}
    mock_connector.execute.return_value = __import__('pandas').DataFrame({"x": [1]})
    mock_store = MagicMock()
    from src.agents.sql_agent import SQLAgent
    agent = SQLAgent(llm=mock_llm, connector=mock_connector, store=mock_store)
    agent.run("c1", "show revenue", [])
    system_prompt = mock_llm.complete.call_args[0][1]
    assert "funnel" not in system_prompt.lower()
    assert "cohort" not in system_prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_sql_agent.py::test_funnel_mode_adds_hint_to_system_prompt tests/agents/test_sql_agent.py::test_cohort_mode_adds_hint_to_system_prompt -v
```

Expected: FAIL with `TypeError: SQLAgent.run() got unexpected keyword argument 'analysis_mode'`

- [ ] **Step 3: Add analysis_mode to SQLAgent**

Replace the `run` method in `src/agents/sql_agent.py`:

```python
    def run(
        self,
        client_id: str,
        request: str,
        context: list[str],
        analysis_mode: str | None = None,
    ) -> AgentResult:
        schema = self._connector.get_schema()
        hint = ""
        if analysis_mode == "funnel":
            hint = (
                "\nFunnel analysis: write a query that computes conversion rates "
                "and drop-off counts for each ordered stage."
            )
        elif analysis_mode == "cohort":
            hint = (
                "\nCohort analysis: write a query that groups users by acquisition "
                "period and computes retention rates by period offset."
            )
        system = (
            "You are a SQL expert. Given a user request, database schema, and business context, "
            "generate a single valid read-only SQL SELECT query. "
            "Return ONLY the SQL query — no explanation, no markdown, no backticks."
            + hint
        )
        user = (
            f"Request: {request}\n\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Business context:\n{chr(10).join(context)}\n\n"
            "Write the SQL SELECT query:"
        )
        sql = self._llm.complete(TaskType.TOOL, system, user).strip()
        try:
            df = self._connector.execute(sql)
            self._store.add(
                client_id,
                [f"Successful SQL for: {request}\nQuery: {sql}"],
                [str(uuid.uuid4())],
                [{"type": "sql_template", "client_id": client_id}],
            )
            return AgentResult(
                agent_name="sql_agent",
                success=True,
                data={
                    "query": sql,
                    "rows": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                },
            )
        except Exception as e:
            return AgentResult(agent_name="sql_agent", success=False, error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_sql_agent.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/sql_agent.py tests/agents/test_sql_agent.py
git commit -m "feat: add analysis_mode to SQLAgent for funnel and cohort SQL hints"
```

---

### Task 5: Orchestrator wiring — anomaly, memory, funnel/cohort

**Files:**
- Modify: `src/orchestrator/orchestrator.py`
- Modify: `tests/orchestrator/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/orchestrator/test_orchestrator.py`:

```python
from unittest.mock import MagicMock
from src.agents.anomaly_agent import AnomalyAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.core.models import (
    AgentResult, Channel, ClarificationState, ClientConfig,
    Request, Response, SkillModule, Tier,
)
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.clarifier import ClarificationChecker
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.chart_agent import ChartAgent


def _make_orc_deps():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Analysis complete.",
    ]
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock(spec=ClarificationChecker)
    state = ClarificationState(original_request=_make_request())
    state.is_resolved = True
    mock_clarifier.check.return_value = state
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT 1", "rows": [{"churn": 0.15}], "columns": ["churn"]},
    )
    return dict(
        llm=mock_llm, retriever=mock_retriever, clarifier=mock_clarifier,
        sql_agent=mock_sql, chart_agent=ChartAgent(),
    )


def _make_request(text="show churn"):
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Ana",
        text=text, timestamp="2026-05-14T00:00:00", client_id="c1",
    )


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ADVANCED,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def test_orchestrator_runs_anomaly_agent_when_skill_enabled():
    deps = _make_orc_deps()
    mock_anomaly = MagicMock()
    mock_anomaly.run.return_value = AgentResult(
        agent_name="anomaly_agent", success=True,
        data={"anomalies": [{"metric": "churn", "value": 0.15, "severity": "critical",
                              "description": "churn breach", "mode": "hard_rule",
                              "expected": None, "operator": ">", "threshold": 0.1}]},
    )
    orc = Orchestrator(**deps, anomaly_agent=mock_anomaly)
    result = orc.process(_make_request(), _make_config(SkillModule.SQL_QUERYING, SkillModule.HARD_RULE_ANOMALY))
    mock_anomaly.run.assert_called_once()
    assert isinstance(result, Response)
    assert len(result.anomalies) == 1


def test_orchestrator_skips_anomaly_when_skill_not_enabled():
    deps = _make_orc_deps()
    mock_anomaly = MagicMock()
    orc = Orchestrator(**deps, anomaly_agent=mock_anomaly)
    orc.process(_make_request(), _make_config(SkillModule.SQL_QUERYING))
    mock_anomaly.run.assert_not_called()


def test_orchestrator_calls_memory_logger():
    deps = _make_orc_deps()
    mock_logger = MagicMock()
    orc = Orchestrator(**deps, memory_logger=mock_logger)
    result = orc.process(_make_request(text="show revenue"), _make_config(SkillModule.SQL_QUERYING))
    mock_logger.log.assert_called_once()
    call_kwargs = mock_logger.log.call_args[1]
    assert call_kwargs["request"].text == "show revenue"


def test_orchestrator_funnel_mode_passes_analysis_mode_to_sql():
    deps = _make_orc_deps()
    deps["llm"].complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": true, "cohort": false}',
        "Funnel analysis complete.",
    ]
    orc = Orchestrator(**deps)
    orc.process(_make_request(text="show funnel"), _make_config(SkillModule.SQL_QUERYING, SkillModule.FUNNEL_ANALYSIS))
    call_args = deps["sql_agent"].run.call_args
    assert call_args[1].get("analysis_mode") == "funnel" or "funnel" in call_args[0]

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/orchestrator/test_orchestrator.py -v -k "anomaly or memory_logger"
```

Expected: FAIL with `TypeError: Orchestrator.__init__() got unexpected keyword argument 'anomaly_agent'`

- [ ] **Step 3: Update Orchestrator**

In `src/orchestrator/orchestrator.py`, update `__init__`:

```python
from src.agents.anomaly_agent import AnomalyAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.core.models import (
    Anomaly, AgentResult, ClarificationState, ClientConfig, Request,
    Response, SkillModule, TaskType,
)

class Orchestrator:
    def __init__(
        self,
        llm: LLMRouter,
        retriever: KnowledgeRetriever,
        clarifier: ClarificationChecker,
        sql_agent: SQLAgent,
        chart_agent: ChartAgent,
        ml_agent: MLAgent | None = None,
        deck_agent: DeckAgent | None = None,
        anomaly_agent: AnomalyAgent | None = None,
        memory_logger: InteractionMemoryLogger | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._clarifier = clarifier
        self._sql_agent = sql_agent
        self._chart_agent = chart_agent
        self._ml_agent = ml_agent
        self._deck_agent = deck_agent
        self._anomaly_agent = anomaly_agent
        self._memory_logger = memory_logger
```

Update `_plan()` — change system prompt and fallback:

```python
    def _plan(self, request: Request, context: list[str]) -> dict:
        system = (
            "Determine which capabilities are needed to answer this data request. "
            "Return JSON only: "
            '{"sql": true/false, "chart": true/false, "ml": true/false, '
            '"deck": true/false, "funnel": true/false, "cohort": true/false}. '
            "Set ml=true for forecast/predict/projection/trend requests. "
            "Set deck=true for slide/deck/presentation/powerpoint requests. "
            "Set funnel=true for funnel or conversion analysis requests. "
            "Set cohort=true for cohort or retention analysis requests."
        )
        user = f"Request: {request.text}\nContext: {chr(10).join(context)}"
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"sql": True, "chart": True, "ml": False, "deck": False,
                    "funnel": False, "cohort": False}
```

Update `process()` — add anomaly dispatch after chart, memory logging at end, and analysis_mode for SQL:

```python
    def process(
        self,
        request: Request,
        config: ClientConfig,
        clarification_state: ClarificationState | None = None,
    ) -> Response | ClarificationState:
        context = self._retriever.search(config.client_id, request.text)
        state = self._clarifier.check(request, context, clarification_state)

        if not state.is_resolved:
            return state

        plan = self._plan(request, context)
        charts: list[bytes] = []
        sql_data: dict | None = None
        sql_queries: list[str] = []
        anomalies: list[dict] = []

        if plan.get("sql") and SkillModule.SQL_QUERYING in config.enabled_skills:
            analysis_mode = None
            if plan.get("funnel") and SkillModule.FUNNEL_ANALYSIS in config.enabled_skills:
                analysis_mode = "funnel"
            elif plan.get("cohort") and SkillModule.COHORT_ANALYSIS in config.enabled_skills:
                analysis_mode = "cohort"
            sql_result = self._sql_agent.run(
                config.client_id, request.text, context, analysis_mode=analysis_mode
            )
            if sql_result.success:
                sql_data = sql_result.data
                if sql_data and sql_data.get("query"):
                    sql_queries.append(sql_data["query"])

                if plan.get("chart") and SkillModule.DATA_VISUALIZATION in config.enabled_skills and sql_data:
                    chart_result = self._chart_agent.run(sql_data)
                    if chart_result.success and chart_result.chart_png:
                        charts.append(chart_result.chart_png)

        # Anomaly detection — runs on sql_data when skill enabled
        anomaly_skill_enabled = (
            SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
            or SkillModule.STATISTICAL_ANOMALY in config.enabled_skills
        )
        if self._anomaly_agent is not None and anomaly_skill_enabled and sql_data:
            if (SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
                    and SkillModule.STATISTICAL_ANOMALY in config.enabled_skills):
                mode = "both"
            elif SkillModule.HARD_RULE_ANOMALY in config.enabled_skills:
                mode = "hard"
            else:
                mode = "statistical"
            anomaly_result = self._anomaly_agent.run(config.client_id, sql_data, mode=mode)
            if anomaly_result.success:
                anomalies = anomaly_result.data.get("anomalies", [])

        # ML Agent
        if (
            plan.get("ml")
            and self._ml_agent is not None
            and SkillModule.MACHINE_LEARNING in config.enabled_skills
            and sql_data
        ):
            ml_result = self._ml_agent.run(config.client_id, request.text, sql_data)
            if ml_result.success and ml_result.chart_png:
                charts.append(ml_result.chart_png)

        text = self._generate_response(request, context, sql_data, state.assumptions)

        # Deck Agent
        deck_pptx: bytes | None = None
        if (
            plan.get("deck")
            and self._deck_agent is not None
            and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
        ):
            sections = [{"heading": "Analysis", "body": text}]
            deck_result = self._deck_agent.run(
                title=request.text[:100], sections=sections, charts=charts
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
        )

        # Interaction memory — always log successful responses
        if self._memory_logger is not None:
            self._memory_logger.log(
                request=request,
                response=response,
                sql_queries=sql_queries,
            )

        return response
```

- [ ] **Step 4: Run full test suite**

```bash
.venv/bin/python -m pytest --tb=short -q
```

Expected: all existing tests PASS plus the new orchestrator tests pass

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/orchestrator.py tests/orchestrator/test_orchestrator.py
git commit -m "feat: wire AnomalyAgent, InteractionMemoryLogger, and funnel/cohort into Orchestrator"
```

---

### Task 6: main.py wiring + integration tests

**Files:**
- Modify: `main.py`
- Create: `tests/test_integration_phase4.py`

- [ ] **Step 1: Update main.py**

Add imports after existing imports:

```python
from src.agents.anomaly_agent import AnomalyAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
```

Add instantiation after `deck_agent = DeckAgent()`:

```python
anomaly_agent = AnomalyAgent(retriever=retriever)
memory_logger = InteractionMemoryLogger(store=store)
```

Update the `Orchestrator(...)` call to add the new params:

```python
orchestrator = Orchestrator(
    llm=llm,
    retriever=retriever,
    clarifier=clarifier,
    sql_agent=None,
    chart_agent=chart_agent,
    ml_agent=ml_agent,
    deck_agent=deck_agent,
    anomaly_agent=anomaly_agent,
    memory_logger=memory_logger,
)
```

- [ ] **Step 2: Write integration tests**

```python
# tests/test_integration_phase4.py
import json
import pytest
from unittest.mock import MagicMock, patch

from src.agents.anomaly_agent import AnomalyAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.knowledge.retriever import KnowledgeRetriever
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.agents.chart_agent import ChartAgent
from src.core.models import (
    Channel, ClientConfig, Request, Response, SkillModule, Tier,
)


def _make_request(text="show churn", client_id="c1"):
    return Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Ana",
        text=text,
        timestamp="2026-05-14T00:00:00",
        client_id=client_id,
    )


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test Corp", tier=Tier.ADVANCED,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_orchestrator(anomaly_agent=None, memory_logger=None):
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Churn rate is elevated at 15%.",
    ]
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock(spec=ClarificationChecker)
    from src.core.models import ClarificationState
    state = ClarificationState(original_request=_make_request())
    state.is_resolved = True
    mock_clarifier.check.return_value = state
    mock_sql = MagicMock()
    from src.core.models import AgentResult
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT churn FROM metrics",
              "rows": [{"churn": 0.15}], "columns": ["churn"]},
    )
    return Orchestrator(
        llm=mock_llm,
        retriever=mock_retriever,
        clarifier=mock_clarifier,
        sql_agent=mock_sql,
        chart_agent=ChartAgent(),
        anomaly_agent=anomaly_agent,
        memory_logger=memory_logger,
    )


def test_anomaly_agent_flags_critical_breach_in_response():
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = [
        json.dumps({"metric": "churn", "operator": ">", "threshold": 0.10, "severity": "critical"})
    ]
    anomaly_agent = AnomalyAgent(retriever=mock_retriever)
    orc = _make_orchestrator(anomaly_agent=anomaly_agent)
    config = _make_config(SkillModule.SQL_QUERYING, SkillModule.HARD_RULE_ANOMALY)
    result = orc.process(_make_request(), config)
    assert isinstance(result, Response)
    assert len(result.anomalies) == 1
    assert result.anomalies[0].severity == "critical"
    assert result.anomalies[0].metric == "churn"


def test_interaction_memory_logger_receives_log_call():
    mock_logger = MagicMock(spec=InteractionMemoryLogger)
    orc = _make_orchestrator(memory_logger=mock_logger)
    config = _make_config(SkillModule.SQL_QUERYING)
    orc.process(_make_request(text="show revenue"), config)
    mock_logger.log.assert_called_once()
    call_kwargs = mock_logger.log.call_args[1]
    assert call_kwargs["request"].text == "show revenue"
    assert "SELECT churn FROM metrics" in call_kwargs["sql_queries"]


def test_no_anomalies_when_skill_not_in_config():
    mock_anomaly = MagicMock(spec=AnomalyAgent)
    orc = _make_orchestrator(anomaly_agent=mock_anomaly)
    config = _make_config(SkillModule.SQL_QUERYING)  # no HARD_RULE_ANOMALY
    result = orc.process(_make_request(), config)
    mock_anomaly.run.assert_not_called()
    assert result.anomalies == []
```

- [ ] **Step 3: Run integration tests**

```bash
.venv/bin/python -m pytest tests/test_integration_phase4.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 4: Run full test suite**

```bash
.venv/bin/python -m pytest --tb=short -q
```

Expected: all tests PASS, count ≥ 109 (92 existing + 17 new)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_integration_phase4.py
git commit -m "feat: wire Phase 4 agents into main.py and add integration tests"
```
