# Phase 6: HypothesisTestingAgent, SegmentationAgent, ABTestingAgent, Storyline Deck

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the storyline-driven DeckAgent path, and build + wire three new statistical agents (HypothesisTestingAgent, SegmentationAgent, ABTestingAgent) into the Orchestrator and main.py.

**Architecture:** Each new agent follows the existing `AgentResult` contract — `run(client_id, request, data) -> AgentResult`. The Orchestrator's `_plan()` LLM call is extended to detect hypothesis/segment/ab_test intent, and three new optional routing blocks are added after the ML block. The DeckAgent's orchestrator path is changed from legacy flat sections to `build_storyline()` + `run(storyline=...)`. JobScheduler is already wired in main.py — no changes needed there.

**Tech Stack:** scipy.stats (already installed), sklearn.cluster.KMeans + sklearn.metrics.silhouette_score + sklearn.preprocessing.StandardScaler (already installed via scikit-learn), matplotlib (already installed), pandas (already installed).

---

## File Structure

**New files:**
- `src/agents/hypothesis_agent.py` — t-test / chi-squared on two-group data
- `src/agents/segmentation_agent.py` — KMeans clustering with auto-k via silhouette
- `src/agents/ab_agent.py` — proportions z-test (binary) or t-test (continuous) A/B evaluation
- `tests/agents/test_hypothesis_agent.py`
- `tests/agents/test_segmentation_agent.py`
- `tests/agents/test_ab_agent.py`
- `tests/test_integration_phase6.py`

**Modified files:**
- `src/orchestrator/orchestrator.py` — deck storyline path (lines ~197-214), `_plan()` prompt, three new routing blocks, three new `__init__` params
- `main.py` — import + instantiate three new agents, pass to Orchestrator

---

### Task 1: DeckAgent storyline path — verify + test

The orchestrator's deck block already calls `build_storyline()` (edited earlier in the session). This task writes tests that confirm the new path works correctly and reverts any stale legacy behavior if needed.

**Files:**
- Modify: `src/orchestrator/orchestrator.py` (verify deck block is correct)
- Modify: `tests/test_integration_phase6.py` (create file)

- [ ] **Step 1: Confirm current deck block in orchestrator**

Read `src/orchestrator/orchestrator.py` lines 197–220. It should look like:

```python
        # Deck Agent — builds PPTX via storyline (LLM) or fallback flat sections
        deck_pptx: bytes | None = None
        if (
            plan.get("deck")
            and self._deck_agent is not None
            and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
        ):
            findings = (
                [f"{r}" for r in (sql_data.get("rows", []) or [])[:5]]
                if sql_data else []
            )
            storyline = self._deck_agent.build_storyline(
                analysis_text=text,
                findings=findings,
                solutions=[],
                recommendation="",
            )
            deck_result = self._deck_agent.run(
                title=request.text[:100], storyline=storyline, charts=charts
            )
            if deck_result.success:
                deck_pptx = deck_result.deck_pptx
```

If it still uses the legacy `sections=[...]` path, replace it with the block above.

- [ ] **Step 2: Write failing integration test for storyline deck**

Create `tests/test_integration_phase6.py`:

```python
# tests/test_integration_phase6.py
import io
import json
from unittest.mock import MagicMock, patch
import pytest
from pptx import Presentation

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.clarifier import ClarificationChecker
from src.agents.deck_agent import DeckAgent
from src.core.models import ClientConfig, Tier, SkillModule, Channel, Request


def _make_llm(plan_json=None, response_text="Analysis complete."):
    mock = MagicMock()
    def complete(task_type, system, user):
        if "which capabilities" in system.lower():
            return plan_json or '{"sql": false, "chart": false, "ml": false, "deck": true, "funnel": false, "cohort": false}'
        if "consulting analyst" in system.lower() or "structure the following" in system.lower():
            return json.dumps({
                "problem_statement": "Revenue dropped.",
                "executive_summary": ["Revenue down 10%"],
                "key_findings": [{"heading": "Drop", "body": "Revenue fell.", "so_what": "Act now.", "chart_index": None}],
                "solutions": [],
                "recommended_solution": "",
                "recommendation_rationale": "",
                "conclusion": "Act now.",
                "next_steps": ["Fix it"],
            })
        if "root-cause" in system.lower() or "deep investigation" in system.lower():
            return "false"
        if "enough information" in system.lower() or "clarif" in system.lower():
            return json.dumps({"resolved": True, "questions": [], "assumptions": []})
        return response_text
    mock.complete.side_effect = complete
    return mock


def _make_config(skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=skills, account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text, client_id="c1"):
    return Request(
        channel=Channel.SLACK, sender_id="u1", sender_name="User",
        text=text, timestamp="2026-05-16T00:00:00Z", client_id=client_id,
    )


def _make_orchestrator(llm, **agents):
    from src.knowledge.retriever import KnowledgeRetriever
    from src.agents.chart_agent import ChartAgent
    from src.agents.sql_agent import SQLAgent
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    clarifier = ClarificationChecker(llm=llm)
    return Orchestrator(
        llm=llm,
        retriever=mock_retriever,
        clarifier=clarifier,
        sql_agent=MagicMock(spec=SQLAgent),
        chart_agent=MagicMock(spec=ChartAgent),
        **agents,
    )


def test_orchestrator_deck_uses_storyline_path():
    llm = _make_llm()
    deck_agent = DeckAgent(llm=llm)
    orch = _make_orchestrator(llm, deck_agent=deck_agent)
    config = _make_config([SkillModule.PRESENTATION_BUILDING])
    result = orch.process(_make_request("Build me a presentation of the analysis"), config)
    assert hasattr(result, "deck_pptx")
    assert result.deck_pptx is not None
    prs = Presentation(io.BytesIO(result.deck_pptx))
    # Storyline path produces title + exec summary + problem + findings + recommendation + conclusion
    assert len(prs.slides) >= 4
```

- [ ] **Step 3: Run test to verify it passes (storyline path already wired)**

```bash
.venv/bin/python -m pytest tests/test_integration_phase6.py::test_orchestrator_deck_uses_storyline_path -v
```

Expected: PASS (the deck block was already updated). If FAIL, fix the orchestrator deck block per Step 1.

- [ ] **Step 4: Commit the orchestrator deck change**

```bash
git add src/orchestrator/orchestrator.py tests/test_integration_phase6.py
git commit -m "feat: wire storyline-driven DeckAgent path in Orchestrator"
```

---

### Task 2: HypothesisTestingAgent

**Files:**
- Create: `src/agents/hypothesis_agent.py`
- Create: `tests/agents/test_hypothesis_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/test_hypothesis_agent.py`:

```python
# tests/agents/test_hypothesis_agent.py
import numpy as np
from unittest.mock import MagicMock

from src.agents.hypothesis_agent import HypothesisAgent


def _make_agent():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Group B shows significantly higher values."
    return HypothesisAgent(llm=mock_llm)


def _two_group_continuous():
    rng = np.random.default_rng(42)
    rows_a = [{"group": "A", "metric": float(v)} for v in rng.normal(10, 1, 30)]
    rows_b = [{"group": "B", "metric": float(v)} for v in rng.normal(15, 1, 30)]
    return {"columns": ["group", "metric"], "rows": rows_a + rows_b}


def _two_group_binary():
    rows = (
        [{"group": "A", "converted": 1} for _ in range(20)] +
        [{"group": "A", "converted": 0} for _ in range(10)] +
        [{"group": "B", "converted": 1} for _ in range(25)] +
        [{"group": "B", "converted": 0} for _ in range(5)]
    )
    return {"columns": ["group", "converted"], "rows": rows}


def test_ttest_detects_significant_difference():
    agent = _make_agent()
    result = agent.run("c1", "Is there a significant difference?", _two_group_continuous())
    assert result.success is True
    assert result.data["test"] == "t_test"
    assert result.data["p_value"] < 0.05
    assert result.data["significant"] is True


def test_returns_statistic_and_groups():
    agent = _make_agent()
    result = agent.run("c1", "Compare groups", _two_group_continuous())
    assert "statistic" in result.data
    assert "groups" in result.data
    assert len(result.data["groups"]) == 2


def test_binary_metric_uses_chi2():
    agent = _make_agent()
    result = agent.run("c1", "Compare conversion rates", _two_group_binary())
    assert result.success is True
    assert result.data["test"] == "chi2"


def test_llm_interpretation_included():
    agent = _make_agent()
    result = agent.run("c1", "Compare groups", _two_group_continuous())
    assert result.data["interpretation"] == "Group B shows significantly higher values."


def test_too_few_rows_returns_error():
    agent = _make_agent()
    data = {"columns": ["group", "metric"], "rows": [{"group": "A", "metric": 1.0}]}
    result = agent.run("c1", "Compare", data)
    assert result.success is False
    assert result.error is not None


def test_no_group_column_returns_error():
    agent = _make_agent()
    data = {
        "columns": ["x", "y"],
        "rows": [{"x": float(i), "y": float(i * 2)} for i in range(10)],
    }
    result = agent.run("c1", "Compare", data)
    assert result.success is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_hypothesis_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.hypothesis_agent'`

- [ ] **Step 3: Implement HypothesisAgent**

Create `src/agents/hypothesis_agent.py`:

```python
import logging
import pandas as pd
from scipy import stats
from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class HypothesisAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            df = pd.DataFrame(data.get("rows", []))
            if df.empty:
                return AgentResult(agent_name="hypothesis_agent", success=False,
                                   error="No data provided")

            cat_cols = [c for c in df.columns
                        if df[c].dtype == object or df[c].nunique() <= 5]
            num_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])]

            group_col = next((c for c in cat_cols if df[c].nunique() == 2), None)
            metric_col = next((c for c in num_cols if c != group_col), None)

            if group_col is None or metric_col is None:
                return AgentResult(
                    agent_name="hypothesis_agent", success=False,
                    error="Need exactly one 2-group column and one numeric metric column",
                )

            groups = df[group_col].unique()
            g1 = df[df[group_col] == groups[0]][metric_col].dropna().values
            g2 = df[df[group_col] == groups[1]][metric_col].dropna().values

            if len(g1) < 2 or len(g2) < 2:
                return AgentResult(
                    agent_name="hypothesis_agent", success=False,
                    error="Each group needs at least 2 observations",
                )

            unique_vals = set(df[metric_col].dropna().unique())
            if unique_vals <= {0, 1}:
                ct = pd.crosstab(df[group_col], df[metric_col])
                stat, p_value, _, _ = stats.chi2_contingency(ct)
                test_name = "chi2"
            else:
                stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
                test_name = "t_test"

            significant = bool(p_value < 0.05)
            interpretation = ""
            if self._llm is not None:
                system = "Interpret this statistical test result in 1-2 sentences for a business audience."
                user = (
                    f"Test: {test_name}, Groups: {groups[0]} vs {groups[1]}, "
                    f"Metric: {metric_col}, p-value: {p_value:.4f}, Significant: {significant}"
                )
                interpretation = self._llm.complete(TaskType.SIMPLE, system, user)

            return AgentResult(
                agent_name="hypothesis_agent",
                success=True,
                data={
                    "test": test_name,
                    "group_col": group_col,
                    "metric_col": metric_col,
                    "groups": list(groups),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": significant,
                    "interpretation": interpretation,
                },
            )
        except Exception as exc:
            _logger.error("HypothesisAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="hypothesis_agent", success=False, error=str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_hypothesis_agent.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/hypothesis_agent.py tests/agents/test_hypothesis_agent.py
git commit -m "feat: add HypothesisTestingAgent (t-test + chi-squared)"
```

---

### Task 3: SegmentationAgent

**Files:**
- Create: `src/agents/segmentation_agent.py`
- Create: `tests/agents/test_segmentation_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/test_segmentation_agent.py`:

```python
# tests/agents/test_segmentation_agent.py
import numpy as np
from unittest.mock import MagicMock

from src.agents.segmentation_agent import SegmentationAgent


def _make_agent():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Segment 0 is low-value, Segment 1 is high-value."
    return SegmentationAgent(llm=mock_llm)


def _clusterable_data(n=20):
    rng = np.random.default_rng(42)
    cluster1 = [{"revenue": float(r), "sessions": float(s)}
                for r, s in zip(rng.normal(100, 5, n), rng.normal(50, 3, n))]
    cluster2 = [{"revenue": float(r), "sessions": float(s)}
                for r, s in zip(rng.normal(500, 10, n), rng.normal(200, 8, n))]
    return {"columns": ["revenue", "sessions"], "rows": cluster1 + cluster2}


def test_segmentation_returns_n_segments():
    agent = _make_agent()
    result = agent.run("c1", "Segment customers", _clusterable_data())
    assert result.success is True
    assert "n_segments" in result.data
    assert result.data["n_segments"] >= 2


def test_segmentation_returns_chart_png():
    agent = _make_agent()
    result = agent.run("c1", "Segment customers", _clusterable_data())
    assert result.chart_png is not None
    assert len(result.chart_png) > 0


def test_segmentation_returns_centroids():
    agent = _make_agent()
    result = agent.run("c1", "Segment", _clusterable_data())
    assert "centroids" in result.data
    assert len(result.data["centroids"]) == result.data["n_segments"]


def test_rows_have_segment_column():
    agent = _make_agent()
    result = agent.run("c1", "Segment", _clusterable_data())
    assert "rows" in result.data
    assert all("segment" in r for r in result.data["rows"])


def test_silhouette_score_in_data():
    agent = _make_agent()
    result = agent.run("c1", "Segment", _clusterable_data())
    assert "silhouette_score" in result.data
    assert isinstance(result.data["silhouette_score"], float)


def test_insufficient_columns_returns_error():
    agent = _make_agent()
    data = {"columns": ["revenue"], "rows": [{"revenue": float(i)} for i in range(10)]}
    result = agent.run("c1", "Segment", data)
    assert result.success is False
    assert result.error is not None


def test_too_few_rows_returns_error():
    agent = _make_agent()
    data = {"columns": ["x", "y"], "rows": [{"x": 1.0, "y": 2.0}]}
    result = agent.run("c1", "Segment", data)
    assert result.success is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_segmentation_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.segmentation_agent'`

- [ ] **Step 3: Implement SegmentationAgent**

Create `src/agents/segmentation_agent.py`:

```python
import io
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class SegmentationAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            df = pd.DataFrame(data.get("rows", []))
            num_df = df.select_dtypes(include=[np.number])

            if num_df.shape[1] < 2:
                return AgentResult(agent_name="segmentation_agent", success=False,
                                   error="Need at least 2 numeric columns for segmentation")
            if len(num_df) < 4:
                return AgentResult(agent_name="segmentation_agent", success=False,
                                   error="Need at least 4 rows for segmentation")

            X = StandardScaler().fit_transform(num_df.values)
            max_k = min(6, len(X) // 2)
            if max_k < 2:
                max_k = 2

            best_k, best_score = 2, -1.0
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
                if score > best_score:
                    best_score, best_k = score, k

            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            df = df.copy()
            df["segment"] = labels

            centroids = {
                f"segment_{seg}": {
                    col: float(num_df[labels == seg][col].mean())
                    for col in num_df.columns
                }
                for seg in range(best_k)
            }

            col1, col2 = num_df.columns[0], num_df.columns[1]
            fig, ax = plt.subplots(figsize=(8, 5))
            for seg in range(best_k):
                mask = labels == seg
                ax.scatter(num_df[mask][col1], num_df[mask][col2],
                           label=f"Segment {seg}", alpha=0.7)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f"Customer Segmentation (k={best_k})")
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            chart_png = buf.read()

            interpretation = ""
            if self._llm is not None:
                system = "Describe these customer segments in 1-2 sentences per segment for a business audience."
                user = f"Segments and their average metrics: {centroids}"
                interpretation = self._llm.complete(TaskType.SIMPLE, system, user)

            return AgentResult(
                agent_name="segmentation_agent",
                success=True,
                chart_png=chart_png,
                data={
                    "n_segments": best_k,
                    "silhouette_score": float(best_score),
                    "centroids": centroids,
                    "interpretation": interpretation,
                    "rows": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                },
            )
        except Exception as exc:
            _logger.error("SegmentationAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="segmentation_agent", success=False, error=str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_segmentation_agent.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/segmentation_agent.py tests/agents/test_segmentation_agent.py
git commit -m "feat: add SegmentationAgent (KMeans auto-k via silhouette)"
```

---

### Task 4: ABTestingAgent

**Files:**
- Create: `src/agents/ab_agent.py`
- Create: `tests/agents/test_ab_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/agents/test_ab_agent.py`:

```python
# tests/agents/test_ab_agent.py
import numpy as np
from unittest.mock import MagicMock

from src.agents.ab_agent import ABTestingAgent


def _make_agent():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Ship the variant."
    return ABTestingAgent(llm=mock_llm)


def _binary_ab_data():
    rows = (
        [{"variant": "control", "converted": 1} for _ in range(100)] +
        [{"variant": "control", "converted": 0} for _ in range(900)] +
        [{"variant": "treatment", "converted": 1} for _ in range(150)] +
        [{"variant": "treatment", "converted": 0} for _ in range(850)]
    )
    return {"columns": ["variant", "converted"], "rows": rows}


def _continuous_ab_data():
    rng = np.random.default_rng(42)
    rows = (
        [{"variant": "control", "revenue": float(v)} for v in rng.normal(50, 5, 100)] +
        [{"variant": "treatment", "revenue": float(v)} for v in rng.normal(58, 5, 100)]
    )
    return {"columns": ["variant", "revenue"], "rows": rows}


def test_binary_ab_returns_proportions_ztest():
    agent = _make_agent()
    result = agent.run("c1", "A/B test results", _binary_ab_data())
    assert result.success is True
    assert result.data["test"] == "proportions_ztest"


def test_continuous_ab_returns_ttest():
    agent = _make_agent()
    result = agent.run("c1", "Compare revenue", _continuous_ab_data())
    assert result.success is True
    assert result.data["test"] == "t_test"


def test_returns_positive_uplift_for_better_variant():
    agent = _make_agent()
    result = agent.run("c1", "A/B results", _binary_ab_data())
    assert "uplift" in result.data
    assert result.data["uplift"] > 0  # treatment (15%) > control (10%)


def test_continuous_significant_result():
    agent = _make_agent()
    result = agent.run("c1", "Revenue test", _continuous_ab_data())
    assert result.data["significant"] is True


def test_returns_control_and_variant_means():
    agent = _make_agent()
    result = agent.run("c1", "A/B test", _binary_ab_data())
    assert "control_mean" in result.data
    assert "variant_mean" in result.data
    assert abs(result.data["control_mean"] - 0.10) < 0.01
    assert abs(result.data["variant_mean"] - 0.15) < 0.01


def test_llm_recommendation_included():
    agent = _make_agent()
    result = agent.run("c1", "A/B test", _binary_ab_data())
    assert result.data["recommendation"] == "Ship the variant."


def test_no_control_group_returns_error():
    agent = _make_agent()
    data = {
        "columns": ["variant", "metric"],
        "rows": [{"variant": "A", "metric": float(i)} for i in range(10)] +
                [{"variant": "B", "metric": float(i)} for i in range(10)],
    }
    result = agent.run("c1", "test", data)
    assert result.success is False
    assert result.error is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_ab_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.ab_agent'`

- [ ] **Step 3: Implement ABTestingAgent**

Create `src/agents/ab_agent.py`:

```python
import logging
import pandas as pd
from scipy import stats
from scipy.stats import proportions_ztest
from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class ABTestingAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            df = pd.DataFrame(data.get("rows", []))

            variant_col = next(
                (c for c in df.columns
                 if df[c].dtype == object
                 and "control" in [str(v).lower() for v in df[c].unique()]),
                None,
            )
            if variant_col is None:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error="No variant column with 'control' group found")

            num_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c]) and c != variant_col]
            if not num_cols:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error="No numeric metric column found")

            metric_col = num_cols[0]
            groups = df[variant_col].unique()
            control_label = next(g for g in groups if str(g).lower() == "control")
            variant_label = next(g for g in groups if str(g).lower() != "control")

            control_data = df[df[variant_col] == control_label][metric_col].dropna().values
            variant_data = df[df[variant_col] == variant_label][metric_col].dropna().values

            if len(control_data) < 2 or len(variant_data) < 2:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error="Each group needs at least 2 observations")

            unique_vals = set(df[metric_col].dropna().unique())
            if unique_vals <= {0, 1}:
                count_c, count_v = int(control_data.sum()), int(variant_data.sum())
                n_c, n_v = len(control_data), len(variant_data)
                stat, p_value = proportions_ztest([count_v, count_c], [n_v, n_c])
                test_name = "proportions_ztest"
                control_mean = count_c / n_c
                variant_mean = count_v / n_v
            else:
                stat, p_value = stats.ttest_ind(variant_data, control_data, equal_var=False)
                test_name = "t_test"
                control_mean = float(control_data.mean())
                variant_mean = float(variant_data.mean())

            uplift = (variant_mean - control_mean) / control_mean if control_mean != 0 else 0.0
            significant = bool(p_value < 0.05)

            recommendation = ""
            if self._llm is not None:
                system = "Give a 1-sentence recommendation on whether to ship the variant based on this A/B test."
                user = (
                    f"Control: {control_mean:.4f}, Variant: {variant_mean:.4f}, "
                    f"Uplift: {uplift:.2%}, p-value: {p_value:.4f}, Significant: {significant}"
                )
                recommendation = self._llm.complete(TaskType.SIMPLE, system, user)

            return AgentResult(
                agent_name="ab_agent",
                success=True,
                data={
                    "test": test_name,
                    "variant_col": variant_col,
                    "metric_col": metric_col,
                    "control_mean": control_mean,
                    "variant_mean": variant_mean,
                    "uplift": uplift,
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": significant,
                    "recommendation": recommendation,
                },
            )
        except Exception as exc:
            _logger.error("ABTestingAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="ab_agent", success=False, error=str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ab_agent.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/ab_agent.py tests/agents/test_ab_agent.py
git commit -m "feat: add ABTestingAgent (proportions z-test + Welch t-test)"
```

---

### Task 5: Wire all new agents into Orchestrator and main.py

**Files:**
- Modify: `src/orchestrator/orchestrator.py`
- Modify: `main.py`

- [ ] **Step 1: Extend Orchestrator `__init__` with three new optional agents**

In `src/orchestrator/orchestrator.py`, add these imports after the existing agent imports:

```python
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.segmentation_agent import SegmentationAgent
from src.agents.ab_agent import ABTestingAgent
```

Extend `__init__` signature (add after `spreadsheet_agent` param):

```python
        hypothesis_agent: "HypothesisAgent | None" = None,
        segmentation_agent: "SegmentationAgent | None" = None,
        ab_agent: "ABTestingAgent | None" = None,
```

Add to `__init__` body (after `self._spreadsheet_agent = spreadsheet_agent`):

```python
        self._hypothesis_agent = hypothesis_agent
        self._segmentation_agent = segmentation_agent
        self._ab_agent = ab_agent
```

- [ ] **Step 2: Extend `_plan()` LLM prompt to detect new intents**

Replace the `system` string in `_plan()`:

```python
        system = (
            "Determine which capabilities are needed to answer this data request. "
            "Return JSON only: "
            '{"sql": true/false, "chart": true/false, "ml": true/false, '
            '"deck": true/false, "funnel": true/false, "cohort": true/false, '
            '"hypothesis": true/false, "segment": true/false, "ab_test": true/false}. '
            "Set ml=true for forecast/predict/projection/trend requests. "
            "Set deck=true for slide/deck/presentation/powerpoint requests. "
            "Set funnel=true for funnel or conversion analysis requests. "
            "Set cohort=true for cohort or retention analysis requests. "
            "Set hypothesis=true for hypothesis test/statistical significance/p-value/compare groups requests. "
            "Set segment=true for customer segmentation/clustering/group customers requests. "
            "Set ab_test=true for A/B test/experiment/variant evaluation requests."
        )

Also update the fallback `return` dict at the end of `_plan()` to include the new keys:

```python
        return {"sql": True, "chart": True, "ml": False, "deck": False,
                "funnel": False, "cohort": False,
                "hypothesis": False, "segment": False, "ab_test": False}
```

- [ ] **Step 3: Add three routing blocks in `process()` after the ML block**

After the ML block (after `if plan.get("ml") ... ml_result ...`), add:

```python
        # Hypothesis testing — runs on sql_data when skill enabled
        if (
            plan.get("hypothesis")
            and self._hypothesis_agent is not None
            and SkillModule.HYPOTHESIS_TESTING in config.enabled_skills
            and sql_data
        ):
            hyp_result = self._hypothesis_agent.run(config.client_id, request.text, sql_data)
            if hyp_result.success:
                sql_data = {**(sql_data or {}), **hyp_result.data}

        # Segmentation — runs on sql_data when skill enabled
        if (
            plan.get("segment")
            and self._segmentation_agent is not None
            and SkillModule.SEGMENTATION in config.enabled_skills
            and sql_data
        ):
            seg_result = self._segmentation_agent.run(config.client_id, request.text, sql_data)
            if seg_result.success:
                if seg_result.chart_png:
                    charts.append(seg_result.chart_png)
                sql_data = {**(sql_data or {}), **seg_result.data}

        # A/B testing — runs on sql_data when skill enabled
        if (
            plan.get("ab_test")
            and self._ab_agent is not None
            and SkillModule.AB_TESTING in config.enabled_skills
            and sql_data
        ):
            ab_result = self._ab_agent.run(config.client_id, request.text, sql_data)
            if ab_result.success:
                sql_data = {**(sql_data or {}), **ab_result.data}
```

- [ ] **Step 4: Wire new agents into main.py**

In `main.py`, add imports after the existing agent imports:

```python
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.segmentation_agent import SegmentationAgent
from src.agents.ab_agent import ABTestingAgent
```

Add instantiation after `spreadsheet_agent = SpreadsheetAgent(llm=llm)`:

```python
hypothesis_agent = HypothesisAgent(llm=llm)
segmentation_agent = SegmentationAgent(llm=llm)
ab_agent = ABTestingAgent(llm=llm)
```

Pass them to the `Orchestrator(...)` call:

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
    analyst_agent=analyst_agent,
    spreadsheet_agent=spreadsheet_agent,
    hypothesis_agent=hypothesis_agent,
    segmentation_agent=segmentation_agent,
    ab_agent=ab_agent,
)
```

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
.venv/bin/python -m pytest --tb=short -q
```

Expected: all existing tests pass (163+).

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/orchestrator.py main.py
git commit -m "feat: wire HypothesisAgent, SegmentationAgent, ABTestingAgent into Orchestrator and main.py"
```

---

### Task 6: Phase 6 integration tests + full suite

**Files:**
- Modify: `tests/test_integration_phase6.py` (add hypothesis, segmentation, ab_test orchestrator routing tests)

- [ ] **Step 1: Add orchestrator routing tests to test_integration_phase6.py**

Append to `tests/test_integration_phase6.py`:

```python
def test_orchestrator_routes_to_hypothesis_agent_when_plan_set():
    from src.agents.hypothesis_agent import HypothesisAgent
    import numpy as np

    plan_json = '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": true, "segment": false, "ab_test": false}'
    llm = _make_llm(plan_json=plan_json)
    mock_hyp = MagicMock(spec=HypothesisAgent)
    mock_hyp.run.return_value = MagicMock(success=True, data={"test": "t_test", "p_value": 0.001, "significant": True, "groups": ["A", "B"], "statistic": 5.2, "group_col": "group", "metric_col": "metric", "interpretation": "Significant."})

    rng = np.random.default_rng(42)
    rows = (
        [{"group": "A", "metric": float(v)} for v in rng.normal(10, 1, 20)] +
        [{"group": "B", "metric": float(v)} for v in rng.normal(15, 1, 20)]
    )
    sql_data = {"columns": ["group", "metric"], "rows": rows, "query": "SELECT ..."}

    mock_sql = MagicMock()
    mock_sql.run.return_value = MagicMock(success=True, data=sql_data)

    orch = _make_orchestrator(llm, hypothesis_agent=mock_hyp)
    orch._sql_agent = mock_sql

    config = _make_config([SkillModule.SQL_QUERYING, SkillModule.HYPOTHESIS_TESTING])
    result = orch.process(_make_request("Is there a significant difference between groups A and B?"), config)
    mock_hyp.run.assert_called_once()


def test_orchestrator_routes_to_segmentation_agent_when_plan_set():
    from src.agents.segmentation_agent import SegmentationAgent
    import numpy as np

    plan_json = '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": true, "ab_test": false}'
    llm = _make_llm(plan_json=plan_json)
    mock_seg = MagicMock(spec=SegmentationAgent)
    mock_seg.run.return_value = MagicMock(success=True, chart_png=b"PNG", data={"n_segments": 2, "centroids": {}, "silhouette_score": 0.7, "rows": [], "columns": [], "interpretation": ""})

    rng = np.random.default_rng(42)
    rows = [{"revenue": float(r), "sessions": float(s)}
            for r, s in zip(rng.normal(100, 10, 20), rng.normal(50, 5, 20))]
    sql_data = {"columns": ["revenue", "sessions"], "rows": rows, "query": "SELECT ..."}

    mock_sql = MagicMock()
    mock_sql.run.return_value = MagicMock(success=True, data=sql_data)

    orch = _make_orchestrator(llm, segmentation_agent=mock_seg)
    orch._sql_agent = mock_sql

    config = _make_config([SkillModule.SQL_QUERYING, SkillModule.SEGMENTATION])
    result = orch.process(_make_request("Segment our customers by revenue and sessions"), config)
    mock_seg.run.assert_called_once()


def test_orchestrator_routes_to_ab_agent_when_plan_set():
    from src.agents.ab_agent import ABTestingAgent

    plan_json = '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": false, "ab_test": true}'
    llm = _make_llm(plan_json=plan_json)
    mock_ab = MagicMock(spec=ABTestingAgent)
    mock_ab.run.return_value = MagicMock(success=True, data={"test": "proportions_ztest", "uplift": 0.05, "significant": True, "control_mean": 0.1, "variant_mean": 0.15, "statistic": 3.1, "p_value": 0.002, "variant_col": "variant", "metric_col": "converted", "recommendation": "Ship."})

    rows = (
        [{"variant": "control", "converted": 1} for _ in range(10)] +
        [{"variant": "control", "converted": 0} for _ in range(90)] +
        [{"variant": "treatment", "converted": 1} for _ in range(15)] +
        [{"variant": "treatment", "converted": 0} for _ in range(85)]
    )
    sql_data = {"columns": ["variant", "converted"], "rows": rows, "query": "SELECT ..."}

    mock_sql = MagicMock()
    mock_sql.run.return_value = MagicMock(success=True, data=sql_data)

    orch = _make_orchestrator(llm, ab_agent=mock_ab)
    orch._sql_agent = mock_sql

    config = _make_config([SkillModule.SQL_QUERYING, SkillModule.AB_TESTING])
    result = orch.process(_make_request("Evaluate our A/B test results"), config)
    mock_ab.run.assert_called_once()
```

- [ ] **Step 2: Run Phase 6 integration tests**

```bash
.venv/bin/python -m pytest tests/test_integration_phase6.py -v
```

Expected: 4 passed (deck storyline + 3 routing tests).

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest --tb=short -q
```

Expected: all tests pass (183+ including Phase 6).

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_phase6.py
git commit -m "test: Phase 6 integration tests — storyline deck, hypothesis, segmentation, A/B routing"
```

---

## Self-Review

**Spec coverage:**
- ✅ DeckAgent storyline path wired in Orchestrator (Task 1)
- ✅ HypothesisTestingAgent — t-test + chi-squared (Task 2)
- ✅ SegmentationAgent — KMeans auto-k silhouette (Task 3)
- ✅ ABTestingAgent — proportions z-test + Welch t-test (Task 4)
- ✅ All three new agents wired into Orchestrator `_plan()` + routing + `__init__` (Task 5)
- ✅ All three new agents instantiated in main.py (Task 5)
- ✅ Integration tests cover orchestrator routing for all new paths (Task 6)
- ✅ JobScheduler already wired in main.py — no changes needed
- ✅ LookerIntegration — deferred (requires external Looker API/SDK, out of scope)

**Placeholder scan:** No TBDs or incomplete sections found.

**Type consistency:**
- All agents use `run(client_id: str, request: str, data: dict) -> AgentResult` — matches existing pattern
- `SkillModule.HYPOTHESIS_TESTING`, `SEGMENTATION`, `AB_TESTING` all exist in `src/core/models.py`
- `_plan()` fallback dict updated to include `"hypothesis": False, "segment": False, "ab_test": False` — must be added in Task 5 Step 2
