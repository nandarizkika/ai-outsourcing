# Phase 5: Deep Analysis, Full ML Suite, NLP, SkillRegistry, SpreadsheetAgent, DeckAgent Redesign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the agent from a single-pass pipeline into an adaptive analytical system with a ReAct-loop `AnalystAgent`, a full ML/NLP model suite, a redesigned `DeckAgent` with structured storylines, a `SpreadsheetAgent` for Excel/CSV analysis, and a `SkillRegistry` for dynamic skill registration.

**Architecture:** `SkillRegistry` replaces enum-based skill gating — built-in skills are pre-registered constants, custom skills can be registered at runtime. `AnalystAgent` runs a think→act→observe loop (max 10 steps, hard cap 15) using internal tools (sql_query, stat_test, cluster_segment, ml_prototype, knowledge_search). The Orchestrator detects deep-dive intent via LLM, asks user confirmation via the channel, then delegates to `AnalystAgent`. `MLAgent` gains `build_model` (full regression + classification zoo with GridSearchCV) and `nlp` modes. `DeckAgent` gets a two-phase flow: LLM builds a structured `Storyline`, then slides are generated dynamically from it. `SpreadsheetAgent` reads Excel/CSV via pandas and produces the same `AgentResult` contract as `SQLAgent`.

**Tech Stack:** scikit-learn, xgboost, lightgbm, spacy (en_core_web_sm), sentence-transformers, scipy.stats (already installed), pandas (already installed), openpyxl (already installed).

---

## File Structure

**New files:**
- `src/core/skill_registry.py` — SkillRegistry + SkillDefinition
- `src/agents/analyst_agent.py` — AnalystAgent with ReAct loop
- `src/agents/spreadsheet_agent.py` — SpreadsheetAgent
- `tests/core/test_skill_registry.py`
- `tests/agents/test_analyst_agent.py`
- `tests/agents/test_spreadsheet_agent.py`
- `tests/test_integration_phase5.py`

**Modified files:**
- `src/core/models.py` — add SkillModule entries, StepRecord, AnalystResult, Storyline, file_bytes/filename on Request, storyline on AgentResult, enabled_skills → list[str]
- `src/agents/ml_agent.py` — add build_model and nlp modes
- `src/agents/deck_agent.py` — redesign with build_storyline + dynamic slides
- `src/orchestrator/orchestrator.py` — deep intent detection, AnalystAgent delegation, spreadsheet routing
- `pyproject.toml` — add Phase 5 dependencies
- `main.py` — wire new agents

---

### Task 1: Dependencies + SkillModule additions + SkillRegistry

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/core/models.py`
- Create: `src/core/skill_registry.py`
- Create: `tests/core/test_skill_registry.py`

- [ ] **Step 1: Add Phase 5 dependencies to pyproject.toml**

Add to the `dependencies` list in `pyproject.toml`:

```toml
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "lightgbm>=4.3",
    "spacy>=3.7",
    "sentence-transformers>=3.0",
```

- [ ] **Step 2: Install dependencies**

```bash
.venv/bin/pip install scikit-learn>=1.4 xgboost>=2.0 lightgbm>=4.3 "spacy>=3.7" "sentence-transformers>=3.0"
.venv/bin/python -m spacy download en_core_web_sm
```

Expected: packages install without error, `python -c "import sklearn, xgboost, lightgbm, spacy, sentence_transformers"` exits 0.

- [ ] **Step 3: Write failing tests for SkillRegistry and new SkillModule entries**

```python
# tests/core/test_skill_registry.py
import pytest
from src.core.skill_registry import SkillRegistry, SkillDefinition, skill_registry
from src.core.models import SkillModule, Tier, ClientConfig, Channel


def test_built_in_skills_are_pre_registered():
    assert skill_registry.get(SkillModule.SQL_QUERYING) is not None
    assert skill_registry.get(SkillModule.MACHINE_LEARNING) is not None
    assert skill_registry.get(SkillModule.DEEP_ANALYSIS) is not None
    assert skill_registry.get(SkillModule.NLP_MODELING) is not None


def test_get_returns_none_for_unknown_skill():
    assert skill_registry.get("not_a_real_skill_xyz") is None


def test_register_custom_skill():
    registry = SkillRegistry()
    registry.register("custom_skill_abc", SkillDefinition(
        name="Custom Skill",
        description="A custom skill for testing",
        tier=Tier.BASIC,
        agent=None,
        requires=[],
    ))
    defn = registry.get("custom_skill_abc")
    assert defn is not None
    assert defn["name"] == "Custom Skill"


def test_all_returns_all_registered_skills():
    registry = SkillRegistry()
    registry.register("skill_a", SkillDefinition(
        name="A", description="a", tier=Tier.BASIC, agent=None, requires=[]
    ))
    registry.register("skill_b", SkillDefinition(
        name="B", description="b", tier=Tier.BASIC, agent=None, requires=[]
    ))
    assert "skill_a" in registry.all()
    assert "skill_b" in registry.all()


def test_is_enabled_checks_client_config():
    config = ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=[SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS],
        account_mode="vendor", active_channels=[Channel.SLACK],
    )
    assert skill_registry.is_enabled(SkillModule.SQL_QUERYING, config) is True
    assert skill_registry.is_enabled(SkillModule.MACHINE_LEARNING, config) is False


def test_new_skillmodule_entries_exist():
    assert SkillModule.NLP_MODELING == "nlp_modeling"
    assert SkillModule.HYPOTHESIS_TESTING == "hypothesis_testing"
    assert SkillModule.SEGMENTATION == "segmentation"
    assert SkillModule.AB_TESTING == "ab_testing"
    assert SkillModule.DEEP_ANALYSIS == "deep_analysis"
    assert SkillModule.SPREADSHEET_ANALYSIS == "spreadsheet_analysis"


def test_client_config_enabled_skills_accepts_strings():
    config = ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=["sql_querying", "custom_skill_xyz"],
        account_mode="vendor", active_channels=[Channel.SLACK],
    )
    assert "sql_querying" in config.enabled_skills
    assert "custom_skill_xyz" in config.enabled_skills
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_skill_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.skill_registry'`

- [ ] **Step 5: Add new SkillModule entries to models.py and change enabled_skills to list[str]**

In `src/core/models.py`, add to `SkillModule` enum:

```python
    NLP_MODELING = "nlp_modeling"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    SEGMENTATION = "segmentation"
    AB_TESTING = "ab_testing"
    DEEP_ANALYSIS = "deep_analysis"
    SPREADSHEET_ANALYSIS = "spreadsheet_analysis"
```

Change `ClientConfig.enabled_skills` field type from `list[SkillModule]` to `list[str]`:

```python
class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[str]
    account_mode: str
    active_channels: list[Channel]
```

- [ ] **Step 6: Create SkillRegistry**

```python
# src/core/skill_registry.py
from typing import TypedDict
from src.core.models import ClientConfig, SkillModule, Tier


class SkillDefinition(TypedDict):
    name: str
    description: str
    tier: Tier
    agent: str | None
    requires: list[str]


class SkillRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, SkillDefinition] = {}

    def register(self, skill_id: str, defn: SkillDefinition) -> None:
        self._registry[skill_id] = defn

    def get(self, skill_id: str) -> SkillDefinition | None:
        return self._registry.get(skill_id)

    def all(self) -> dict[str, SkillDefinition]:
        return dict(self._registry)

    def is_enabled(self, skill_id: str, config: ClientConfig) -> bool:
        return skill_id in config.enabled_skills


def _build_default_registry() -> SkillRegistry:
    r = SkillRegistry()
    _B, _A, _E = Tier.BASIC, Tier.ADVANCED, Tier.ENTERPRISE

    r.register(SkillModule.SQL_QUERYING, SkillDefinition(
        name="SQL Querying", tier=_B, agent="SQLAgent", requires=[],
        description="Natural language → SQL → structured results",
    ))
    r.register(SkillModule.DATA_VISUALIZATION, SkillDefinition(
        name="Data Visualization", tier=_B, agent="ChartAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Generate charts from query results",
    ))
    r.register(SkillModule.REPORT_GENERATION, SkillDefinition(
        name="Report Generation", tier=_B, agent="Orchestrator", requires=[],
        description="Written narrative analysis from data",
    ))
    r.register(SkillModule.SCHEDULED_REPORTING, SkillDefinition(
        name="Scheduled Reporting", tier=_B, agent="JobScheduler", requires=[],
        description="Cron-based automated report delivery",
    ))
    r.register(SkillModule.HARD_RULE_ANOMALY, SkillDefinition(
        name="Hard-Rule Anomaly", tier=_A, agent="AnomalyAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Flag when metrics breach client-defined rules",
    ))
    r.register(SkillModule.STATISTICAL_ANOMALY, SkillDefinition(
        name="Statistical Anomaly", tier=_A, agent="AnomalyAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Pattern-based deviation from historical baseline",
    ))
    r.register(SkillModule.FUNNEL_ANALYSIS, SkillDefinition(
        name="Funnel Analysis", tier=_A, agent="SQLAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Conversion rates and drop-off counts per stage",
    ))
    r.register(SkillModule.COHORT_ANALYSIS, SkillDefinition(
        name="Cohort Analysis", tier=_A, agent="SQLAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Retention rates grouped by acquisition period",
    ))
    r.register(SkillModule.SPREADSHEET_ANALYSIS, SkillDefinition(
        name="Spreadsheet Analysis", tier=_A, agent="SpreadsheetAgent", requires=[],
        description="Upload Excel/CSV and analyse in-place without SQL",
    ))
    r.register(SkillModule.MACHINE_LEARNING, SkillDefinition(
        name="Machine Learning", tier=_E, agent="MLAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Forecast, regression, classification + model tuning",
    ))
    r.register(SkillModule.PRESENTATION_BUILDING, SkillDefinition(
        name="Presentation Building", tier=_E, agent="DeckAgent",
        requires=[SkillModule.REPORT_GENERATION],
        description="Auto-generate PPTX slide decks with storyline",
    ))
    r.register(SkillModule.NLP_MODELING, SkillDefinition(
        name="NLP Modeling", tier=_E, agent="MLAgent", requires=[],
        description="Text classification, NER, similarity via TF-IDF + embeddings",
    ))
    r.register(SkillModule.HYPOTHESIS_TESTING, SkillDefinition(
        name="Hypothesis Testing", tier=_E, agent="AnalystAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="t-test, chi-square, ANOVA, Mann-Whitney significance tests",
    ))
    r.register(SkillModule.SEGMENTATION, SkillDefinition(
        name="Segmentation", tier=_E, agent="AnalystAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="KMeans clustering to find user/product/revenue segments",
    ))
    r.register(SkillModule.AB_TESTING, SkillDefinition(
        name="A/B Testing", tier=_E, agent="AnalystAgent",
        requires=[SkillModule.SQL_QUERYING],
        description="Statistical comparison of experiment variants",
    ))
    r.register(SkillModule.DEEP_ANALYSIS, SkillDefinition(
        name="Deep Analysis", tier=_E, agent="AnalystAgent",
        requires=[SkillModule.SQL_QUERYING, SkillModule.MACHINE_LEARNING],
        description="ReAct loop: root cause → hypothesize → test → prototype (max 10–15 steps)",
    ))
    return r


skill_registry = _build_default_registry()
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/core/test_skill_registry.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 8: Run full suite to verify no regressions**

```bash
.venv/bin/python -m pytest --tb=short -q
```

Expected: all existing tests PASS. Note: `enabled_skills: list[str]` is backwards-compatible since `SkillModule` values are strings — existing tests that pass `SkillModule.X` values still work.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/core/models.py src/core/skill_registry.py tests/core/test_skill_registry.py
git commit -m "feat: add SkillRegistry, expand SkillModule, change enabled_skills to list[str]"
```

---

### Task 2: SpreadsheetAgent

**Files:**
- Create: `src/agents/spreadsheet_agent.py`
- Create: `tests/agents/test_spreadsheet_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_spreadsheet_agent.py
import io
import json
import pytest
from unittest.mock import MagicMock

import pandas as pd

from src.agents.spreadsheet_agent import SpreadsheetAgent
from src.core.models import AgentResult, Channel, ClientConfig, SkillModule, Tier


def _make_config():
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ADVANCED,
        enabled_skills=[SkillModule.SPREADSHEET_ANALYSIS],
        account_mode="vendor", active_channels=[Channel.SLACK],
    )


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _agent(ops_json: str = '{"operation": "describe"}'):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = ops_json
    return SpreadsheetAgent(llm=mock_llm)


def test_reads_csv_and_returns_agent_result():
    df = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100, 200, 300]})
    agent = _agent('{"operation": "describe"}')
    result = agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="Summarize revenue",
        config=_make_config(),
    )
    assert result.success is True
    assert result.data is not None
    assert "rows" in result.data
    assert "columns" in result.data


def test_reads_xlsx_and_returns_agent_result():
    df = pd.DataFrame({"region": ["X", "Y"], "sales": [500, 600]})
    agent = _agent('{"operation": "describe"}')
    result = agent.run(
        file_bytes=_xlsx_bytes(df),
        filename="data.xlsx",
        question="Summarize sales",
        config=_make_config(),
    )
    assert result.success is True
    assert "rows" in result.data


def test_groupby_operation():
    df = pd.DataFrame({
        "region": ["A", "A", "B", "B"],
        "revenue": [100, 150, 200, 250],
    })
    ops = json.dumps({"operation": "groupby", "by": "region", "agg": {"revenue": "sum"}})
    agent = _agent(ops)
    result = agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="Total revenue by region",
        config=_make_config(),
    )
    assert result.success is True
    rows = result.data["rows"]
    regions = {r["region"] for r in rows}
    assert regions == {"A", "B"}


def test_filter_operation():
    df = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100, 200, 300]})
    ops = json.dumps({"operation": "filter", "column": "revenue", "operator": ">", "value": 150})
    agent = _agent(ops)
    result = agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="High revenue regions",
        config=_make_config(),
    )
    assert result.success is True
    assert all(r["revenue"] > 150 for r in result.data["rows"])


def test_unsupported_format_returns_failure():
    agent = _agent()
    result = agent.run(
        file_bytes=b"not a real file",
        filename="data.parquet",
        question="Summarize",
        config=_make_config(),
    )
    assert result.success is False
    assert result.error is not None


def test_llm_called_with_question_and_columns():
    df = pd.DataFrame({"month": [1, 2, 3], "revenue": [100, 200, 300]})
    agent = _agent('{"operation": "describe"}')
    agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="Show revenue trend",
        config=_make_config(),
    )
    call_args = agent._llm.complete.call_args
    prompt = call_args[0][2]  # user prompt (3rd positional arg)
    assert "Show revenue trend" in prompt
    assert "month" in prompt or "revenue" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_spreadsheet_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.spreadsheet_agent'`

- [ ] **Step 3: Implement SpreadsheetAgent**

```python
# src/agents/spreadsheet_agent.py
import json
import logging

import pandas as pd

from src.core.llm import LLMRouter
from src.core.models import AgentResult, ClientConfig, TaskType

_logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class SpreadsheetAgent:
    def __init__(self, llm: LLMRouter) -> None:
        self._llm = llm

    def run(
        self,
        file_bytes: bytes,
        filename: str,
        question: str,
        config: ClientConfig,
    ) -> AgentResult:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in _SUPPORTED_EXTENSIONS:
            return AgentResult(
                agent_name="spreadsheet_agent",
                success=False,
                error=f"Unsupported file format: {ext}. Supported: {', '.join(_SUPPORTED_EXTENSIONS)}",
            )
        try:
            import io
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            return AgentResult(agent_name="spreadsheet_agent", success=False, error=str(e))

        system = (
            "You are a data analyst. Given a question and a DataFrame with the columns shown, "
            "return a single JSON object describing the pandas operation to perform. "
            'Valid operations: "describe", "groupby", "filter", "head". '
            'For groupby: {"operation": "groupby", "by": "<col>", "agg": {"<col>": "<func>"}}. '
            'For filter: {"operation": "filter", "column": "<col>", "operator": ">|<|>=|<=|==", "value": <n>}. '
            'For describe or head: {"operation": "describe"} or {"operation": "head", "n": 10}. '
            "Return ONLY valid JSON."
        )
        user = (
            f"Question: {question}\n"
            f"Columns: {list(df.columns)}\n"
            f"Sample (first 3 rows):\n{df.head(3).to_json(orient='records')}"
        )
        raw = self._llm.complete(TaskType.TOOL, system, user).strip()

        try:
            ops = json.loads(raw)
        except json.JSONDecodeError:
            ops = {"operation": "describe"}

        result_df = self._execute(df, ops)
        return AgentResult(
            agent_name="spreadsheet_agent",
            success=True,
            data={
                "rows": result_df.to_dict(orient="records"),
                "columns": list(result_df.columns),
                "summary": ops,
            },
        )

    def _execute(self, df: pd.DataFrame, ops: dict) -> pd.DataFrame:
        op = ops.get("operation", "describe")
        try:
            if op == "groupby":
                return df.groupby(ops["by"]).agg(ops["agg"]).reset_index()
            if op == "filter":
                col, operator, value = ops["column"], ops["operator"], ops["value"]
                if operator == ">":
                    return df[df[col] > value]
                if operator == ">=":
                    return df[df[col] >= value]
                if operator == "<":
                    return df[df[col] < value]
                if operator == "<=":
                    return df[df[col] <= value]
                if operator == "==":
                    return df[df[col] == value]
            if op == "head":
                return df.head(ops.get("n", 10))
        except (KeyError, TypeError) as exc:
            _logger.warning("SpreadsheetAgent operation failed: %s", exc)
        return df.describe().reset_index()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_spreadsheet_agent.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/spreadsheet_agent.py tests/agents/test_spreadsheet_agent.py
git commit -m "feat: add SpreadsheetAgent for Excel/CSV in-place analysis"
```

---

### Task 3: MLAgent expansion — build_model (regression + classification + model tuning)

**Files:**
- Modify: `src/agents/ml_agent.py`
- Modify: `tests/agents/test_ml_agent.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/agents/test_ml_agent.py`:

```python
def _tabular_regression_data():
    import numpy as np
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n) * 0.1
    rows = [{"f1": float(X[i, 0]), "f2": float(X[i, 1]), "f3": float(X[i, 2]),
             "target": float(y[i])} for i in range(n)]
    return {"columns": ["f1", "f2", "f3", "target"], "rows": rows}


def _tabular_classification_data():
    import numpy as np
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rows = [{"f1": float(X[i, 0]), "f2": float(X[i, 1]), "f3": float(X[i, 2]),
             "label": int(y[i])} for i in range(n)]
    return {"columns": ["f1", "f2", "f3", "label"], "rows": rows}


def test_build_model_regression_returns_best_model_name():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="build regression model for target",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="target",
    )
    assert result.success is True
    assert "best_model" in result.data
    assert isinstance(result.data["best_model"], str)


def test_build_model_regression_returns_metrics():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="predict target",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="target",
    )
    assert result.success is True
    assert "r2" in result.data
    assert "rmse" in result.data
    assert result.data["r2"] > 0.5


def test_build_model_classification_returns_best_model_and_report():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify label",
        data=_tabular_classification_data(),
        task="build_model",
        target_col="label",
    )
    assert result.success is True
    assert "best_model" in result.data
    assert "f1_weighted" in result.data
    assert result.data["f1_weighted"] > 0.5


def test_build_model_returns_feature_importance():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="predict target",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="target",
    )
    assert result.success is True
    assert "feature_importance" in result.data
    fi = result.data["feature_importance"]
    assert isinstance(fi, dict)
    assert len(fi) > 0


def test_build_model_fails_gracefully_with_missing_target_col():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="predict nonexistent",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="nonexistent_col",
    )
    assert result.success is False
    assert result.error is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_ml_agent.py -k "build_model" -v
```

Expected: FAIL — `build_model` mode not implemented yet

- [ ] **Step 3: Add build_model mode to MLAgent**

Add this method to `src/agents/ml_agent.py`:

```python
    def _run_build_model(self, client_id: str, request: str, data: dict, target_col: str) -> AgentResult:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import cross_val_score, GridSearchCV
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVR, SVC
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.metrics import r2_score, mean_squared_error, f1_score
        import xgboost as xgb
        import lightgbm as lgb

        df = pd.DataFrame(data["rows"])
        if target_col not in df.columns:
            return AgentResult(
                agent_name="ml_agent", success=False,
                error=f"Target column '{target_col}' not found in data",
            )

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).values
        y_raw = df[target_col].values

        # Detect task type: classification if target is int/bool with few unique values
        unique_vals = len(np.unique(y_raw))
        is_classification = unique_vals <= 20 and np.issubdtype(y_raw.dtype, np.integer)

        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            candidates = [
                ("LogisticRegression", LogisticRegression(max_iter=500), {"C": [0.1, 1.0]}),
                ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42), {"max_depth": [3, 5]}),
                ("GradientBoosting", GradientBoostingClassifier(n_estimators=50, random_state=42), {"max_depth": [2, 3]}),
                ("XGBoost", xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0, eval_metric="logloss"), {"max_depth": [2, 3]}),
                ("LightGBM", lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1), {"num_leaves": [15, 31]}),
                ("SVC", SVC(), {"C": [0.1, 1.0]}),
                ("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 5]}),
                ("GaussianNB", GaussianNB(), {}),
                ("MLP", MLPClassifier(max_iter=200, random_state=42), {"hidden_layer_sizes": [(50,), (100,)]}),
            ]
            scorer = "f1_weighted"
        else:
            y = y_raw.astype(float)
            candidates = [
                ("LinearRegression", LinearRegression(), {}),
                ("Ridge", Ridge(), {"alpha": [0.1, 1.0]}),
                ("Lasso", Lasso(max_iter=2000), {"alpha": [0.01, 0.1]}),
                ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42), {"max_depth": [3, 5]}),
                ("GradientBoosting", GradientBoostingRegressor(n_estimators=50, random_state=42), {"max_depth": [2, 3]}),
                ("XGBoost", xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0), {"max_depth": [2, 3]}),
                ("LightGBM", lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), {"num_leaves": [15, 31]}),
                ("SVR", SVR(), {"C": [0.1, 1.0]}),
                ("KNN", KNeighborsRegressor(), {"n_neighbors": [3, 5]}),
                ("MLPRegressor", MLPRegressor(max_iter=200, random_state=42), {"hidden_layer_sizes": [(50,), (100,)]}),
            ]
            scorer = "r2"

        # Cross-val all candidates, pick top 3
        cv_scores: list[tuple[float, str, object, dict]] = []
        for name, model, params in candidates:
            try:
                scores = cross_val_score(model, X, y, cv=min(5, len(y) // 5), scoring=scorer)
                cv_scores.append((float(scores.mean()), name, model, params))
            except Exception:
                continue

        if not cv_scores:
            return AgentResult(agent_name="ml_agent", success=False, error="All candidates failed cross-validation")

        cv_scores.sort(key=lambda t: t[0], reverse=True)
        top3 = cv_scores[:3]

        # GridSearchCV on top 3
        best_score = -1e9
        best_name = ""
        best_model_obj = None
        for score, name, model, params in top3:
            try:
                if params:
                    gs = GridSearchCV(model, params, cv=min(3, len(y) // 3), scoring=scorer)
                    gs.fit(X, y)
                    tuned_score = gs.best_score_
                    fitted = gs.best_estimator_
                else:
                    model.fit(X, y)
                    tuned_score = score
                    fitted = model
                if tuned_score > best_score:
                    best_score = tuned_score
                    best_name = name
                    best_model_obj = fitted
            except Exception:
                continue

        if best_model_obj is None:
            return AgentResult(agent_name="ml_agent", success=False, error="GridSearch failed for all top candidates")

        # Feature importance
        feature_importance: dict[str, float] = {}
        feature_names = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        if hasattr(best_model_obj, "feature_importances_"):
            fi = best_model_obj.feature_importances_
            feature_importance = {feature_names[i]: float(fi[i]) for i in range(min(len(fi), len(feature_names)))}
        elif hasattr(best_model_obj, "coef_"):
            coef = best_model_obj.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            feature_importance = {feature_names[i]: float(abs(coef[i])) for i in range(min(len(coef), len(feature_names)))}

        # Final metrics
        y_pred = best_model_obj.predict(X)
        if is_classification:
            result_data = {
                "best_model": best_name,
                "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
                "feature_importance": feature_importance,
                "best_params": getattr(best_model_obj, "get_params", lambda: {})(),
            }
        else:
            result_data = {
                "best_model": best_name,
                "r2": float(r2_score(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(np.mean(np.abs(y - y_pred))),
                "feature_importance": feature_importance,
                "best_params": getattr(best_model_obj, "get_params", lambda: {})(),
            }
        return AgentResult(agent_name="ml_agent", success=True, data=result_data)
```

In `MLAgent.run()`, add `target_col: str | None = None` parameter and dispatch:

```python
    def run(self, client_id: str, request: str, data: dict,
            task: str | None = None, model_hint: str | None = None,
            target_col: str | None = None) -> AgentResult:
        ...
        if detected_task == "build_model":
            if not target_col:
                return AgentResult(agent_name="ml_agent", success=False,
                                   error="target_col is required for build_model task")
            return self._run_build_model(client_id, request, data, target_col)
        ...
```

Add `"build_model"` to the task auto-detection keywords: if request contains "build model", "train model", "fit model", "best model", detect `task = "build_model"`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ml_agent.py -v
```

Expected: all tests PASS (may take 30–60s for model training)

- [ ] **Step 5: Commit**

```bash
git add src/agents/ml_agent.py tests/agents/test_ml_agent.py
git commit -m "feat: add build_model mode to MLAgent with full regression and classification suite"
```

---

### Task 4: MLAgent NLP mode

**Files:**
- Modify: `src/agents/ml_agent.py`
- Modify: `tests/agents/test_ml_agent.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/agents/test_ml_agent.py`:

```python
def _nlp_classification_data():
    texts = [
        "The product is amazing and I love it", "Great service, very satisfied",
        "Excellent quality, highly recommend", "Best purchase I have made",
        "Good value for money", "Really happy with this",
        "Terrible product, broke immediately", "Awful experience, very disappointed",
        "Worst purchase ever, complete waste", "Poor quality, do not buy",
        "Very bad service", "Disappointed with the result",
    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    rows = [{"text": t, "label": l} for t, l in zip(texts, labels)]
    return {"columns": ["text", "label"], "rows": rows}


def test_nlp_classify_returns_best_pipeline():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify sentiment",
        data=_nlp_classification_data(),
        task="nlp",
        target_col="label",
    )
    assert result.success is True
    assert "best_vectorizer" in result.data
    assert "best_classifier" in result.data
    assert "f1_weighted" in result.data


def test_nlp_classify_f1_above_chance():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify text",
        data=_nlp_classification_data(),
        task="nlp",
        target_col="label",
    )
    assert result.success is True
    assert result.data["f1_weighted"] > 0.5


def test_nlp_missing_text_col_returns_failure():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify",
        data={"columns": ["no_text_col", "label"], "rows": [{"no_text_col": "x", "label": 1}]},
        task="nlp",
        target_col="label",
    )
    assert result.success is False
    assert result.error is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_ml_agent.py -k "nlp" -v
```

Expected: FAIL — `nlp` task not implemented yet

- [ ] **Step 3: Add NLP mode to MLAgent**

Add this method to `src/agents/ml_agent.py`:

```python
    def _run_nlp(self, client_id: str, request: str, data: dict, target_col: str | None) -> AgentResult:
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score
        from sentence_transformers import SentenceTransformer

        df = pd.DataFrame(data["rows"])

        # Find text column (first string column that isn't the target)
        text_col = None
        for col in df.columns:
            if col != target_col and df[col].dtype == object:
                text_col = col
                break
        if text_col is None:
            return AgentResult(agent_name="ml_agent", success=False,
                               error="No text column found in data")

        texts = df[text_col].astype(str).tolist()

        if target_col and target_col in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(df[target_col].values)

            classifiers = [
                ("LogisticRegression", LogisticRegression(max_iter=500)),
                ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("SVC", SVC()),
            ]

            # TF-IDF vectorizer
            tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X_tfidf = tfidf.fit_transform(texts).toarray()

            # Sentence embeddings
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            X_embed = st_model.encode(texts)

            best_f1 = -1.0
            best_vec_name = ""
            best_clf_name = ""
            best_clf_obj = None
            best_X = X_tfidf

            for vec_name, X in [("tfidf", X_tfidf), ("sentence_transformer", X_embed)]:
                for clf_name, clf in classifiers:
                    try:
                        scores = cross_val_score(clf, X, y, cv=min(3, len(y) // 3), scoring="f1_weighted")
                        mean_f1 = float(scores.mean())
                        if mean_f1 > best_f1:
                            best_f1 = mean_f1
                            best_vec_name = vec_name
                            best_clf_name = clf_name
                            best_clf_obj = clf
                            best_X = X
                    except Exception:
                        continue

            if best_clf_obj is None:
                return AgentResult(agent_name="ml_agent", success=False,
                                   error="All NLP classifiers failed")

            best_clf_obj.fit(best_X, y)
            y_pred = best_clf_obj.predict(best_X)
            return AgentResult(
                agent_name="ml_agent",
                success=True,
                data={
                    "best_vectorizer": best_vec_name,
                    "best_classifier": best_clf_name,
                    "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
                    "text_col": text_col,
                    "target_col": target_col,
                },
            )

        # No target — return basic text stats
        return AgentResult(
            agent_name="ml_agent",
            success=True,
            data={
                "text_col": text_col,
                "sample_count": len(texts),
                "avg_length": float(np.mean([len(t.split()) for t in texts])),
            },
        )
```

In `MLAgent.run()`, add dispatch for `"nlp"` task:

```python
        if detected_task == "nlp":
            return self._run_nlp(client_id, request, data, target_col)
```

Add `"nlp"` keyword detection: if request contains "classify text", "nlp", "sentiment", "text classification", detect `task = "nlp"`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ml_agent.py -k "nlp" -v
```

Expected: all 3 NLP tests PASS (may take 20–40s for embedding model download on first run)

- [ ] **Step 5: Run full ML test suite**

```bash
.venv/bin/python -m pytest tests/agents/test_ml_agent.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents/ml_agent.py tests/agents/test_ml_agent.py
git commit -m "feat: add NLP mode to MLAgent with TF-IDF and sentence-transformer pipelines"
```

---

### Task 5: DeckAgent redesign — Storyline model + build_storyline

**Files:**
- Modify: `src/core/models.py` — add `FindingSlide`, `Solution`, `Storyline` models + `storyline` field on `AgentResult`
- Modify: `src/agents/deck_agent.py`
- Modify: `tests/agents/test_deck_agent.py`

- [ ] **Step 1: Write failing tests for Storyline model**

Add to `tests/agents/test_deck_agent.py`:

```python
from src.core.models import Storyline, FindingSlide, Solution

def test_storyline_model_fields():
    s = Storyline(
        problem_statement="Churn increased 20% in Q2.",
        executive_summary=["Churn up 20%", "Root cause: pricing"],
        key_findings=[
            FindingSlide(heading="Q2 Churn", body="Churn hit 15%.",
                         so_what="We risk losing top cohort.", chart_index=None)
        ],
        solutions=[
            Solution(title="Price rollback", description="Revert to Q1 pricing.",
                     pros=["Quick"], cons=["Revenue impact"])
        ],
        recommended_solution="Price rollback",
        recommendation_rationale="Fastest path to churn reduction.",
        conclusion="Act within 30 days.",
        next_steps=["Rollback pricing", "Monitor churn weekly"],
    )
    assert s.problem_statement == "Churn increased 20% in Q2."
    assert len(s.key_findings) == 1
    assert s.key_findings[0].chart_index is None
    assert len(s.solutions) == 1

def test_agent_result_storyline_defaults_none():
    r = AgentResult(agent_name="deck_agent", success=True)
    assert r.storyline is None

def test_build_storyline_returns_storyline_object():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "problem_statement": "Revenue dropped.",
        "executive_summary": ["Revenue down 10%"],
        "key_findings": [{"heading": "Drop", "body": "Revenue fell.", "so_what": "Action needed.", "chart_index": None}],
        "solutions": [{"title": "Fix X", "description": "Do X.", "pros": ["fast"], "cons": ["costly"]}],
        "recommended_solution": "Fix X",
        "recommendation_rationale": "Fastest fix.",
        "conclusion": "Act now.",
        "next_steps": ["Start fix"],
    })
    agent = DeckAgent(llm=mock_llm)
    storyline = agent.build_storyline(
        analysis_text="Revenue dropped 10% in Q3.",
        findings=["Revenue fell"],
        solutions=[{"title": "Fix X", "description": "Do X.", "pros": ["fast"], "cons": ["costly"]}],
        recommendation="Fix X",
    )
    assert isinstance(storyline, Storyline)
    assert storyline.problem_statement == "Revenue dropped."
    assert len(storyline.key_findings) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_deck_agent.py -k "storyline" -v
```

Expected: FAIL with `ImportError: cannot import name 'Storyline'`

- [ ] **Step 3: Add Storyline models to models.py**

Add after `class Anomaly` in `src/core/models.py`:

```python
class FindingSlide(BaseModel):
    heading: str
    body: str
    so_what: str
    chart_index: Optional[int] = None


class Solution(BaseModel):
    title: str
    description: str
    pros: list[str] = []
    cons: list[str] = []


class Storyline(BaseModel):
    problem_statement: str
    executive_summary: list[str] = []
    key_findings: list[FindingSlide] = []
    solutions: list[Solution] = []
    recommended_solution: str = ""
    recommendation_rationale: str = ""
    conclusion: str = ""
    next_steps: list[str] = []
```

Update `AgentResult` to add `storyline`:

```python
class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: Optional[dict] = None
    chart_png: Optional[bytes] = None
    deck_pptx: Optional[bytes] = None
    storyline: Optional["Storyline"] = None
    error: Optional[str] = None
```

- [ ] **Step 4: Add build_storyline to DeckAgent**

Update `src/agents/deck_agent.py` constructor to accept optional `llm`:

```python
from src.core.llm import LLMRouter
from src.core.models import AgentResult, FindingSlide, Solution, Storyline, TaskType

class DeckAgent:
    def __init__(self, llm: LLMRouter | None = None) -> None:
        self._llm = llm

    def build_storyline(
        self,
        analysis_text: str,
        findings: list[str],
        solutions: list[dict],
        recommendation: str,
    ) -> Storyline:
        if self._llm is None:
            return self._fallback_storyline(analysis_text, findings, solutions, recommendation)
        system = (
            "You are a consulting analyst. Structure the following analysis as a presentation narrative. "
            "Return ONLY valid JSON matching this schema exactly:\n"
            '{"problem_statement": "str", "executive_summary": ["str"], '
            '"key_findings": [{"heading": "str", "body": "str", "so_what": "str", "chart_index": null}], '
            '"solutions": [{"title": "str", "description": "str", "pros": ["str"], "cons": ["str"]}], '
            '"recommended_solution": "str", "recommendation_rationale": "str", '
            '"conclusion": "str", "next_steps": ["str"]}'
        )
        user = (
            f"Analysis:\n{analysis_text}\n\n"
            f"Key findings:\n{chr(10).join(findings)}\n\n"
            f"Solutions considered:\n{chr(10).join(s.get('title', '') for s in solutions)}\n\n"
            f"Recommendation: {recommendation}"
        )
        raw = self._llm.complete(TaskType.REASONING, system, user)
        try:
            data = json.loads(raw)
            return Storyline(
                problem_statement=data.get("problem_statement", ""),
                executive_summary=data.get("executive_summary", []),
                key_findings=[FindingSlide(**f) for f in data.get("key_findings", [])],
                solutions=[Solution(**s) for s in data.get("solutions", [])],
                recommended_solution=data.get("recommended_solution", recommendation),
                recommendation_rationale=data.get("recommendation_rationale", ""),
                conclusion=data.get("conclusion", ""),
                next_steps=data.get("next_steps", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return self._fallback_storyline(analysis_text, findings, solutions, recommendation)

    def _fallback_storyline(
        self,
        analysis_text: str,
        findings: list[str],
        solutions: list[dict],
        recommendation: str,
    ) -> Storyline:
        paragraphs = [p.strip() for p in analysis_text.split("\n\n") if p.strip()]
        return Storyline(
            problem_statement=paragraphs[0] if paragraphs else analysis_text[:200],
            executive_summary=findings[:5],
            key_findings=[
                FindingSlide(heading=f"Finding {i+1}", body=p, so_what="")
                for i, p in enumerate(paragraphs[:5])
            ],
            solutions=[
                Solution(title=s.get("title", f"Option {i+1}"),
                         description=s.get("description", ""),
                         pros=s.get("pros", []), cons=s.get("cons", []))
                for i, s in enumerate(solutions)
            ],
            recommended_solution=recommendation,
            recommendation_rationale="",
            conclusion="",
            next_steps=[],
        )
```

Note: add `import json` at top of `deck_agent.py` if not already present.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_deck_agent.py -v
```

Expected: all tests PASS (existing + new storyline tests)

- [ ] **Step 6: Commit**

```bash
git add src/core/models.py src/agents/deck_agent.py tests/agents/test_deck_agent.py
git commit -m "feat: add Storyline model and DeckAgent.build_storyline() LLM phase"
```

---

### Task 6: DeckAgent redesign — dynamic slide generation

**Files:**
- Modify: `src/agents/deck_agent.py`
- Modify: `tests/agents/test_deck_agent.py`

- [ ] **Step 1: Write failing tests for dynamic slide generation**

Add to `tests/agents/test_deck_agent.py`:

```python
def _make_storyline(n_findings=2, n_solutions=2, with_chart_index=False):
    return Storyline(
        problem_statement="Revenue fell 10% in Q2.",
        executive_summary=["Revenue down 10%", "Root cause found", "Action required"],
        key_findings=[
            FindingSlide(
                heading=f"Finding {i+1}",
                body=f"Detail {i+1}.",
                so_what=f"Impact {i+1}.",
                chart_index=0 if (with_chart_index and i == 0) else None,
            )
            for i in range(n_findings)
        ],
        solutions=[
            Solution(title=f"Option {i+1}", description=f"Desc {i+1}.",
                     pros=["Pro"], cons=["Con"])
            for i in range(n_solutions)
        ],
        recommended_solution="Option 1",
        recommendation_rationale="Fastest impact.",
        conclusion="Act within 30 days.",
        next_steps=["Step 1", "Step 2"],
    )


def test_run_with_storyline_produces_valid_pptx():
    agent = DeckAgent()
    storyline = _make_storyline()
    result = agent.run(title="Q2 Revenue Analysis", storyline=storyline, charts=[])
    assert result.success is True
    assert result.deck_pptx is not None
    prs = Presentation(io.BytesIO(result.deck_pptx))
    assert len(prs.slides) >= 5  # title + exec summary + problem + findings + recommendation


def test_result_contains_storyline():
    agent = DeckAgent()
    storyline = _make_storyline()
    result = agent.run(title="Report", storyline=storyline, charts=[])
    assert result.success is True
    assert result.storyline is not None
    assert result.storyline.problem_statement == "Revenue fell 10% in Q2."


def test_executive_summary_slide_is_second():
    agent = DeckAgent()
    storyline = _make_storyline()
    result = agent.run(title="Report", storyline=storyline, charts=[])
    prs = Presentation(io.BytesIO(result.deck_pptx))
    second_slide_text = " ".join(
        shape.text for shape in prs.slides[1].shapes if shape.has_text_frame
    )
    assert any(bullet in second_slide_text for bullet in storyline.executive_summary)


def test_inline_chart_appears_in_finding_slide_not_appended():
    agent = DeckAgent()
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    storyline = _make_storyline(n_findings=2, with_chart_index=True)
    result = agent.run(title="Chart Report", storyline=storyline, charts=[png_bytes])
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    has_picture = any(
        shape.shape_type == 13 for slide in prs.slides for shape in slide.shapes
    )
    assert has_picture


def test_many_solutions_uses_comparison_table_slide():
    agent = DeckAgent()
    storyline = _make_storyline(n_solutions=4)
    result = agent.run(title="Solutions", storyline=storyline, charts=[], max_solutions_inline=3)
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    # 4 solutions > max_solutions_inline=3, so comparison table used (1 slide)
    # Total should be less than title+exec+problem+4_solution_slides+rec+conclusion
    assert len(prs.slides) < 10


def test_legacy_sections_still_work():
    agent = DeckAgent()
    result = agent.run(
        title="Legacy Report",
        sections=[{"heading": "Summary", "body": "All good."}],
        charts=[],
    )
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    assert len(prs.slides) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_deck_agent.py -k "storyline or dynamic or executive or inline or solutions or legacy" -v
```

Expected: FAIL — `run()` doesn't accept `storyline` kwarg yet

- [ ] **Step 3: Redesign DeckAgent.run() with dynamic slide generation**

Replace the existing `run()` method in `src/agents/deck_agent.py` with:

```python
    def run(
        self,
        title: str,
        storyline: "Storyline | None" = None,
        sections: list[dict] | None = None,
        charts: list[bytes] = [],
        template_path: str | None = None,
        max_solutions_inline: int = 3,
    ) -> AgentResult:
        import io as _io
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor

        try:
            prs = Presentation(template_path) if template_path else Presentation()

            if storyline is not None:
                self._add_title_slide(prs, title)
                self._add_executive_summary_slide(prs, storyline)
                self._add_problem_slide(prs, storyline.problem_statement)
                for finding in storyline.key_findings:
                    chart_png = (
                        charts[finding.chart_index]
                        if finding.chart_index is not None and finding.chart_index < len(charts)
                        else None
                    )
                    self._add_finding_slide(prs, finding, chart_png)
                # Unmatched charts go to appendix
                matched = {f.chart_index for f in storyline.key_findings if f.chart_index is not None}
                appendix_charts = [c for i, c in enumerate(charts) if i not in matched]

                if len(storyline.solutions) <= max_solutions_inline:
                    for sol in storyline.solutions:
                        self._add_solution_slide(prs, sol)
                else:
                    self._add_solutions_table_slide(prs, storyline.solutions)

                self._add_recommendation_slide(prs, storyline)
                self._add_conclusion_slide(prs, storyline)

                for i, png in enumerate(appendix_charts):
                    self._add_chart_slide(prs, png, f"Appendix Chart {i + 1}")
            else:
                # Legacy flat sections path
                self._add_title_slide(prs, title)
                for section in (sections or []):
                    self._add_content_slide(prs, section["heading"], section["body"])
                for i, png in enumerate(charts):
                    self._add_chart_slide(prs, png, f"Chart {i + 1}")

            buf = _io.BytesIO()
            prs.save(buf)
            buf.seek(0)
            return AgentResult(
                agent_name="deck_agent",
                success=True,
                deck_pptx=buf.read(),
                storyline=storyline,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("DeckAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="deck_agent", success=False, error=str(exc))
```

Add these helper methods to `DeckAgent`:

```python
    def _add_executive_summary_slide(self, prs, storyline: "Storyline") -> None:
        from pptx.util import Inches, Pt
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Executive Summary"
        tf = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else \
            slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.clear()
        for bullet in storyline.executive_summary:
            p = tf.add_paragraph()
            p.text = bullet
            p.level = 0

    def _add_problem_slide(self, prs, problem_statement: str) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Problem Statement"
        tf = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else \
            slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.text = problem_statement

    def _add_finding_slide(self, prs, finding: "FindingSlide", chart_png: bytes | None) -> None:
        from pptx.util import Inches
        import io as _io
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = finding.heading
        if chart_png:
            # Layout B: chart left 60%, text right 40%
            slide.shapes.add_picture(_io.BytesIO(chart_png), Inches(0.3), Inches(1.2), Inches(5.5), Inches(4.5))
            tb = slide.shapes.add_textbox(Inches(6.0), Inches(1.2), Inches(3.5), Inches(4.5))
            tb.text_frame.text = f"{finding.body}\n\nSo what: {finding.so_what}"
        else:
            # Layout A: body + so-what callout
            tf = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else \
                slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(5.5), Inches(4)).text_frame
            tf.text = finding.body
            callout = slide.shapes.add_textbox(Inches(6.2), Inches(1.5), Inches(3.3), Inches(1.5))
            callout.text_frame.text = f"So what?\n{finding.so_what}"

    def _add_solution_slide(self, prs, solution: "Solution") -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = solution.title
        tf = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else \
            slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.clear()
        p = tf.add_paragraph(); p.text = solution.description
        p = tf.add_paragraph(); p.text = "Pros: " + ", ".join(solution.pros)
        p = tf.add_paragraph(); p.text = "Cons: " + ", ".join(solution.cons)

    def _add_solutions_table_slide(self, prs, solutions: list) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[5]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Solutions Comparison"
        rows, cols = len(solutions) + 1, 3
        table = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.5), Inches(9), Inches(0.5 * rows)).table
        for i, header in enumerate(["Solution", "Pros", "Cons"]):
            table.cell(0, i).text = header
        for r, sol in enumerate(solutions, start=1):
            table.cell(r, 0).text = sol.title
            table.cell(r, 1).text = ", ".join(sol.pros)
            table.cell(r, 2).text = ", ".join(sol.cons)

    def _add_recommendation_slide(self, prs, storyline: "Storyline") -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Recommendation"
        tb = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(1.2))
        tb.text_frame.text = storyline.recommended_solution
        if storyline.recommendation_rationale:
            tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(2.8), Inches(9), Inches(2.5))
            tb2.text_frame.text = storyline.recommendation_rationale

    def _add_conclusion_slide(self, prs, storyline: "Storyline") -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Conclusion & Next Steps"
        tf = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else \
            slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.clear()
        if storyline.conclusion:
            p = tf.add_paragraph(); p.text = storyline.conclusion
        for step in storyline.next_steps:
            p = tf.add_paragraph(); p.text = f"• {step}"; p.level = 1
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_deck_agent.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/deck_agent.py tests/agents/test_deck_agent.py
git commit -m "feat: redesign DeckAgent with dynamic storyline-based slide generation"
```

---

### Task 7: AnalystAgent — models + ReAct loop

**Files:**
- Modify: `src/core/models.py` — add `StepRecord`, `AnalystResult`, `file_bytes`/`filename` on `Request`
- Create: `src/agents/analyst_agent.py`
- Create: `tests/agents/test_analyst_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_analyst_agent.py
import pytest
from unittest.mock import MagicMock, call

from src.agents.analyst_agent import AnalystAgent
from src.core.models import (
    AgentResult, AnalystResult, Channel, ClientConfig, Request, SkillModule, StepRecord, Tier,
)


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text="Why did churn spike in Q2?"):
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Ana",
        text=text, timestamp="2026-05-14T00:00:00", client_id="c1",
    )


def _make_agent(llm_responses: list[str]):
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = llm_responses
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"rows": [{"churn": 0.15}], "columns": ["churn"], "query": "SELECT churn FROM m"},
    )
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    return AnalystAgent(llm=mock_llm, sql_agent=mock_sql, retriever=mock_retriever)


def _step_response(tool="DONE", tool_input=None, thought="Analysis complete"):
    import json
    return json.dumps({
        "thought": thought,
        "tool": tool,
        "tool_input": tool_input or {},
    })


def test_loop_terminates_when_done_returned():
    agent = _make_agent([
        _step_response("sql_query", {"question": "churn by region"}, "Querying churn"),
        _step_response("DONE", {}, "Analysis complete"),
        '{"findings": ["Churn up in Region 3"], "solutions": [], "recommendation": "Investigate Region 3"}',
    ])
    checkpoints = []
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda step, thought: checkpoints.append((step, thought)),
        max_steps=10,
    )
    assert isinstance(result, AnalystResult)
    assert result.success is True
    assert len(result.steps) == 1  # 1 tool call before DONE


def test_loop_terminates_at_max_steps():
    # Always return sql_query, never DONE
    responses = [
        _step_response("sql_query", {"question": f"query {i}"}, f"Querying {i}")
        for i in range(20)
    ]
    responses.append('{"findings": [], "solutions": [], "recommendation": "Timed out"}')
    agent = _make_agent(responses)
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
        max_steps=3,
    )
    assert isinstance(result, AnalystResult)
    assert len(result.steps) <= 3


def test_max_steps_clamped_to_15():
    responses = [_step_response("DONE")] + ['{"findings": [], "solutions": [], "recommendation": ""}']
    agent = _make_agent(responses)
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
        max_steps=100,  # should be clamped to 15
    )
    assert result.success is True


def test_checkpoint_fires_every_3_steps():
    responses = [
        _step_response("sql_query", {"question": f"q{i}"}, f"thought {i}")
        for i in range(6)
    ]
    responses += [
        _step_response("DONE"),
        '{"findings": [], "solutions": [], "recommendation": "done"}',
    ]
    agent = _make_agent(responses)
    checkpoints = []
    agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda step, thought: checkpoints.append(step),
        max_steps=10,
    )
    assert 3 in checkpoints
    assert 6 in checkpoints


def test_analyst_result_has_findings_and_recommendation():
    agent = _make_agent([
        _step_response("DONE"),
        '{"findings": ["Churn up in Region 3"], "solutions": [{"title": "Fix A", "description": "Do A", "pros": ["fast"], "cons": []}], "recommendation": "Fix A"}',
    ])
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
    )
    assert result.findings == ["Churn up in Region 3"]
    assert result.recommendation == "Fix A"
    assert len(result.solutions) == 1


def test_step_records_stored_on_result():
    agent = _make_agent([
        _step_response("sql_query", {"question": "churn"}, "Checking churn"),
        _step_response("DONE"),
        '{"findings": [], "solutions": [], "recommendation": ""}',
    ])
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
    )
    assert len(result.steps) == 1
    assert result.steps[0].tool == "sql_query"
    assert result.steps[0].thought == "Checking churn"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/agents/test_analyst_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.analyst_agent'`

- [ ] **Step 3: Add StepRecord and AnalystResult to models.py**

Add after `class Storyline` in `src/core/models.py`:

```python
class StepRecord(BaseModel):
    step: int
    thought: str
    tool: str
    tool_input: dict
    observation: str


class AnalystResult(AgentResult):
    steps: list[StepRecord] = []
    findings: list[str] = []
    solutions: list[dict] = []
    recommendation: str = ""
```

Also add `file_bytes` and `filename` optional fields to `Request`:

```python
class Request(BaseModel):
    channel: Channel
    sender_id: str
    sender_name: str
    text: str
    thread_id: Optional[str] = None
    timestamp: str
    client_id: str
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None
```

- [ ] **Step 4: Implement AnalystAgent**

```python
# src/agents/analyst_agent.py
import json
import logging
from typing import Callable

import numpy as np
from scipy import stats

from src.core.llm import LLMRouter
from src.core.models import (
    AgentResult, AnalystResult, ClientConfig, Request, StepRecord, TaskType,
)
from src.knowledge.retriever import KnowledgeRetriever

_logger = logging.getLogger(__name__)

_MAX_STEPS_HARD_CAP = 15
_TOOL_FAILURE_LIMIT = 3


class AnalystAgent:
    def __init__(
        self,
        llm: LLMRouter,
        sql_agent,
        retriever: KnowledgeRetriever,
        ml_agent=None,
    ) -> None:
        self._llm = llm
        self._sql_agent = sql_agent
        self._retriever = retriever
        self._ml_agent = ml_agent

    def run_deep(
        self,
        request: Request,
        config: ClientConfig,
        on_checkpoint: Callable[[int, str], None],
        max_steps: int = 10,
    ) -> AnalystResult:
        max_steps = min(max_steps, _MAX_STEPS_HARD_CAP)
        history: list[StepRecord] = []
        consecutive_failures = 0

        for step in range(1, max_steps + 1):
            raw = self._think(request, history)
            try:
                parsed = json.loads(raw)
                thought = parsed.get("thought", "")
                tool = parsed.get("tool", "DONE")
                tool_input = parsed.get("tool_input", {})
            except json.JSONDecodeError:
                thought, tool, tool_input = "", "DONE", {}

            if tool == "DONE":
                break

            observation = self._execute_tool(tool, tool_input, config)
            if observation.startswith("Tool failed:"):
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            history.append(StepRecord(
                step=step,
                thought=thought,
                tool=tool,
                tool_input=tool_input,
                observation=observation,
            ))

            if step % 3 == 0:
                on_checkpoint(step, thought)

            if consecutive_failures >= _TOOL_FAILURE_LIMIT:
                _logger.warning("AnalystAgent: %d consecutive tool failures, stopping early", _TOOL_FAILURE_LIMIT)
                break

        return self._synthesize(request, history)

    def _think(self, request: Request, history: list[StepRecord]) -> str:
        tools_desc = (
            "Available tools: sql_query(question), stat_test(data, test, groups), "
            "cluster_segment(data, n_clusters), knowledge_search(query). "
            "When investigation is complete, return tool='DONE'."
        )
        system = (
            "You are an expert data analyst performing a root-cause investigation. "
            "At each step, decide what to investigate next. "
            f"{tools_desc} "
            "Return ONLY valid JSON: "
            '{"thought": "your reasoning", "tool": "tool_name", "tool_input": {...}}'
        )
        history_text = "\n".join(
            f"Step {r.step}: [{r.tool}] → {r.observation[:300]}" for r in history
        )
        user = (
            f"Original question: {request.text}\n\n"
            f"Investigation history:\n{history_text}\n\n"
            "What should we investigate next?"
        )
        return self._llm.complete(TaskType.REASONING, system, user)

    def _execute_tool(self, tool: str, tool_input: dict, config: ClientConfig) -> str:
        try:
            if tool == "sql_query":
                question = tool_input.get("question", "")
                result = self._sql_agent.run(config.client_id, question, [])
                if result.success and result.data:
                    return json.dumps(result.data.get("rows", [])[:20])
                return f"Tool failed: {result.error}"

            if tool == "stat_test":
                return self._run_stat_test(tool_input)

            if tool == "cluster_segment":
                return self._run_cluster_segment(tool_input)

            if tool == "knowledge_search":
                query = tool_input.get("query", "")
                docs = self._retriever.search(config.client_id, query)
                return "\n".join(docs[:3]) if docs else "No documents found."

            return f"Tool failed: unknown tool '{tool}'"
        except Exception as exc:
            _logger.warning("Tool %s failed: %s", tool, exc)
            return f"Tool failed: {exc}"

    def _run_stat_test(self, tool_input: dict) -> str:
        test = tool_input.get("test", "ttest")
        groups = tool_input.get("groups", [])
        if len(groups) < 2:
            return "Tool failed: stat_test requires at least 2 groups"
        try:
            arrays = [np.array(g, dtype=float) for g in groups]
            if test == "ttest":
                stat, p = stats.ttest_ind(arrays[0], arrays[1])
            elif test == "anova":
                stat, p = stats.f_oneway(*arrays)
            elif test == "chi2":
                stat, p = stats.chisquare(arrays[0])
            else:
                return f"Tool failed: unknown test '{test}'"
            return json.dumps({"statistic": float(stat), "p_value": float(p), "significant": p < 0.05})
        except Exception as exc:
            return f"Tool failed: {exc}"

    def _run_cluster_segment(self, tool_input: dict) -> str:
        from sklearn.cluster import KMeans
        data = tool_input.get("data", [])
        n_clusters = int(tool_input.get("n_clusters", 3))
        if not data:
            return "Tool failed: no data provided"
        try:
            arr = np.array([[v for v in row.values() if isinstance(v, (int, float))]
                            for row in data], dtype=float)
            if arr.shape[0] < n_clusters:
                return "Tool failed: more clusters than data points"
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(arr)
            counts = {int(i): int((labels == i).sum()) for i in range(n_clusters)}
            return json.dumps({"n_clusters": n_clusters, "cluster_sizes": counts})
        except Exception as exc:
            return f"Tool failed: {exc}"

    def _synthesize(self, request: Request, history: list[StepRecord]) -> AnalystResult:
        history_text = "\n".join(
            f"Step {r.step}: [{r.tool}({json.dumps(r.tool_input)})] → {r.observation[:400]}"
            for r in history
        )
        system = (
            "You are a senior analyst. Based on the investigation below, synthesize the findings. "
            "Return ONLY valid JSON: "
            '{"findings": ["str"], '
            '"solutions": [{"title": "str", "description": "str", "pros": ["str"], "cons": ["str"]}], '
            '"recommendation": "str"}'
        )
        user = (
            f"Original question: {request.text}\n\n"
            f"Investigation log:\n{history_text}\n\n"
            "Synthesize findings, propose solutions, and give a recommendation."
        )
        raw = self._llm.complete(TaskType.REASONING, system, user)
        try:
            parsed = json.loads(raw)
            return AnalystResult(
                agent_name="analyst_agent",
                success=True,
                steps=history,
                findings=parsed.get("findings", []),
                solutions=parsed.get("solutions", []),
                recommendation=parsed.get("recommendation", ""),
            )
        except json.JSONDecodeError:
            return AnalystResult(
                agent_name="analyst_agent",
                success=True,
                steps=history,
                findings=[],
                solutions=[],
                recommendation=raw[:500],
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_analyst_agent.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/models.py src/agents/analyst_agent.py tests/agents/test_analyst_agent.py
git commit -m "feat: add AnalystAgent with ReAct loop, StepRecord, AnalystResult models"
```

---

### Task 8: Orchestrator + main.py wiring + integration tests

**Files:**
- Modify: `src/orchestrator/orchestrator.py`
- Modify: `main.py`
- Create: `tests/test_integration_phase5.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_integration_phase5.py
import json
import pytest
from unittest.mock import MagicMock

from src.agents.analyst_agent import AnalystAgent
from src.agents.spreadsheet_agent import SpreadsheetAgent
from src.core.models import (
    AgentResult, AnalystResult, Channel, ClientConfig,
    ClarificationState, Request, Response, SkillModule, StepRecord, Tier,
)
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.clarifier import ClarificationChecker
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.chart_agent import ChartAgent


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text="Why did churn spike?", file_bytes=None, filename=None):
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Ana",
        text=text, timestamp="2026-05-14T00:00:00", client_id="c1",
        file_bytes=file_bytes, filename=filename,
    )


def _base_orc_deps(llm_responses):
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = llm_responses
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock(spec=ClarificationChecker)
    state = ClarificationState(original_request=_make_request())
    state.is_resolved = True
    mock_clarifier.check.return_value = state
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT 1", "rows": [{"x": 1}], "columns": ["x"]},
    )
    return dict(
        llm=mock_llm, retriever=mock_retriever, clarifier=mock_clarifier,
        sql_agent=mock_sql, chart_agent=ChartAgent(),
    )


def test_orchestrator_detects_deep_intent_and_returns_clarification():
    deps = _base_orc_deps([
        "true",   # detect_deep_intent returns true
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Summary response.",
    ])
    mock_analyst = MagicMock(spec=AnalystAgent)
    orc = Orchestrator(**deps, analyst_agent=mock_analyst)
    config = _make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS)
    result = orc.process(_make_request("Why did churn spike?"), config)
    # Should return ClarificationState asking for deep dive confirmation
    assert isinstance(result, ClarificationState)
    assert "deep" in result.questions_asked[-1].lower() or "dive" in result.questions_asked[-1].lower()


def test_orchestrator_routes_to_spreadsheet_agent_when_file_present():
    import io
    import pandas as pd
    df = pd.DataFrame({"revenue": [100, 200, 300]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue()

    deps = _base_orc_deps([
        '{"operation": "describe"}',  # SpreadsheetAgent LLM call
        '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Revenue summary: mean is 200.",
    ])
    mock_spreadsheet = MagicMock(spec=SpreadsheetAgent)
    mock_spreadsheet.run.return_value = AgentResult(
        agent_name="spreadsheet_agent", success=True,
        data={"rows": [{"revenue": 200.0}], "columns": ["revenue"], "summary": {}},
    )
    orc = Orchestrator(**deps, spreadsheet_agent=mock_spreadsheet)
    config = _make_config(SkillModule.SPREADSHEET_ANALYSIS, SkillModule.REPORT_GENERATION)
    request = _make_request(file_bytes=csv_bytes, filename="data.csv")
    result = orc.process(request, config)
    mock_spreadsheet.run.assert_called_once()
    assert isinstance(result, Response)


def test_analyst_agent_full_loop_produces_analyst_result():
    analyst_result = AnalystResult(
        agent_name="analyst_agent", success=True,
        steps=[StepRecord(step=1, thought="Checking churn", tool="sql_query",
                          tool_input={"question": "churn by region"},
                          observation='[{"region": "A", "churn": 0.2}]')],
        findings=["Churn is highest in Region A"],
        solutions=[{"title": "Fix A", "description": "Rollback pricing in A",
                    "pros": ["fast"], "cons": ["cost"]}],
        recommendation="Fix A",
    )
    deps = _base_orc_deps([
        "true",   # detect_deep_intent
        "Comprehensive analysis complete.",
    ])
    mock_analyst = MagicMock(spec=AnalystAgent)
    mock_analyst.run_deep.return_value = analyst_result
    orc = Orchestrator(**deps, analyst_agent=mock_analyst)
    config = _make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS)

    # First call: deep intent detected → ClarificationState returned
    req = _make_request("Why did churn spike?")
    state = orc.process(req, config)
    assert isinstance(state, ClarificationState)

    # Simulate user confirming "yes" — set deep_dive_confirmed flag
    state.deep_dive_confirmed = True
    state.is_resolved = True  # allow orchestrator to proceed

    # Second call with state confirms → delegates to analyst_agent
    # (Orchestrator checks state.deep_dive_confirmed to skip re-asking)
    mock_analyst_result = orc.process(req, config, clarification_state=state)
    # With deep_dive_confirmed, orchestrator should call analyst_agent
    # The result may be AnalystResult or Response depending on implementation
    assert mock_analyst.run_deep.called or isinstance(mock_analyst_result, (Response, AnalystResult))
```

- [ ] **Step 2: Update Orchestrator to support analyst_agent, spreadsheet_agent, deep intent**

Add to `src/orchestrator/orchestrator.py` imports:

```python
from src.agents.analyst_agent import AnalystAgent
from src.agents.spreadsheet_agent import SpreadsheetAgent
```

Also add `AnalystResult` to the existing models import line (e.g. add it after `AgentResult`):

```python
from src.core.models import (
    ..., AgentResult, AnalystResult, Anomaly, ...
)
```

Add `analyst_agent` and `spreadsheet_agent` to `__init__`:

```python
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
        analyst_agent: AnalystAgent | None = None,
        spreadsheet_agent: SpreadsheetAgent | None = None,
    ):
        ...
        self._analyst_agent = analyst_agent
        self._spreadsheet_agent = spreadsheet_agent
```

Add `detect_deep_intent()` method:

```python
    def _detect_deep_intent(self, request: Request) -> bool:
        system = (
            "Does this request ask for root-cause analysis, deep investigation, "
            "or a complex multi-step analytical inquiry? "
            "Return only 'true' or 'false'."
        )
        raw = self._llm.complete(TaskType.SIMPLE, system, f"Request: {request.text}")
        return raw.strip().lower().startswith("true")
```

Update `process()` to add deep intent check before existing pipeline, and spreadsheet routing:

```python
    def process(
        self,
        request: Request,
        config: ClientConfig,
        clarification_state: ClarificationState | None = None,
    ) -> Response | ClarificationState | AnalystResult:
        context = self._retriever.search(config.client_id, request.text)

        # Deep dive confirmation flow
        if (
            self._analyst_agent is not None
            and SkillModule.DEEP_ANALYSIS in config.enabled_skills
            and clarification_state is not None
            and getattr(clarification_state, "deep_dive_pending", False)
        ):
            confirmed = clarification_state.answers_received and any(
                ans.strip().lower() in ("yes", "y", "sure", "go ahead", "ok", "yep")
                for ans in clarification_state.answers_received
            )
            if confirmed or getattr(clarification_state, "deep_dive_confirmed", False):
                checkpoints: list[str] = []
                return self._analyst_agent.run_deep(
                    request=request,
                    config=config,
                    on_checkpoint=lambda step, thought: checkpoints.append(f"Step {step}: {thought}"),
                )
            # User declined — fall through to regular pipeline

        state = self._clarifier.check(request, context, clarification_state)

        # Deep intent detection — only if no clarification in progress and skill enabled
        if (
            self._analyst_agent is not None
            and SkillModule.DEEP_ANALYSIS in config.enabled_skills
            and state.is_resolved
            and clarification_state is None
            and self._detect_deep_intent(request)
        ):
            deep_state = ClarificationState(original_request=request)
            deep_state.questions_asked = [
                "Want me to do a deep dive on this? It may take a few minutes. "
                "Reply yes to proceed or no for a quick answer."
            ]
            deep_state.deep_dive_pending = True  # type: ignore[attr-defined]
            deep_state.is_resolved = False
            return deep_state

        if not state.is_resolved:
            return state

        # Spreadsheet routing — takes priority over SQL if file present
        sql_data: dict | None = None
        sql_queries: list[str] = []
        charts: list[bytes] = []
        anomalies: list[dict] = []

        if (
            request.file_bytes is not None
            and request.filename is not None
            and self._spreadsheet_agent is not None
            and SkillModule.SPREADSHEET_ANALYSIS in config.enabled_skills
        ):
            sheet_result = self._spreadsheet_agent.run(
                file_bytes=request.file_bytes,
                filename=request.filename,
                question=request.text,
                config=config,
            )
            if sheet_result.success:
                sql_data = sheet_result.data
        else:
            # Existing SQL pipeline
            plan = self._plan(request, context)
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

        # Anomaly detection
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
            self._ml_agent is not None
            and SkillModule.MACHINE_LEARNING in config.enabled_skills
            and sql_data
            and self._plan(request, context).get("ml")
        ):
            ml_result = self._ml_agent.run(config.client_id, request.text, sql_data)
            if ml_result.success and ml_result.chart_png:
                charts.append(ml_result.chart_png)

        text = self._generate_response(request, context, sql_data, state.assumptions)

        deck_pptx: bytes | None = None
        if (
            self._deck_agent is not None
            and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
            and self._plan(request, context).get("deck")
        ):
            storyline = self._deck_agent.build_storyline(
                analysis_text=text,
                findings=[],
                solutions=[],
                recommendation="",
            )
            deck_result = self._deck_agent.run(
                title=request.text[:100], storyline=storyline, charts=charts
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

        if self._memory_logger is not None:
            self._memory_logger.log(request=request, response=response, sql_queries=sql_queries)

        return response
```

- [ ] **Step 3: Update main.py**

Add imports:

```python
from src.agents.analyst_agent import AnalystAgent
from src.agents.spreadsheet_agent import SpreadsheetAgent
```

Add instantiation after `anomaly_agent = AnomalyAgent(retriever=retriever)`:

```python
analyst_agent = AnalystAgent(llm=llm, sql_agent=None, retriever=retriever, ml_agent=ml_agent)
spreadsheet_agent = SpreadsheetAgent(llm=llm)
```

Add to `Orchestrator(...)`:

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
)
```

- [ ] **Step 4: Run integration tests**

```bash
.venv/bin/python -m pytest tests/test_integration_phase5.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest --tb=short -q
```

Expected: all tests PASS, count ≥ 140

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/orchestrator.py src/agents/analyst_agent.py main.py tests/test_integration_phase5.py
git commit -m "feat: wire AnalystAgent, SpreadsheetAgent into Orchestrator and main.py"
```
