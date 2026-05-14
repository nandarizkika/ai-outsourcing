# Phase 4 & Phase 5 Implementation Design

## Goal

Phase 4 delivers anomaly detection and interaction memory. Phase 5 delivers deep investigation capability, a full ML/NLP model suite, redesigned presentation generation, spreadsheet analysis, and a dynamic skill registry — transforming the agent from a single-pass pipeline into an adaptive analytical system.

## Architecture

**Phase 4** adds two independent subsystems on top of the existing single-pass pipeline: an `AnomalyAgent` (hard-rule + statistical detection) and an `InteractionMemoryLogger` (append-only log of every request and response into ChromaDB). It also activates `FUNNEL_ANALYSIS` and `COHORT_ANALYSIS` as prompt-template extensions of `SQLAgent` — no new agent needed.

**Phase 5** introduces an agentic loop via `AnalystAgent` (ReAct pattern), expands `MLAgent` to a full model zoo with hyperparameter tuning and NLP, redesigns `DeckAgent` to generate structured storylines before producing slides, adds `SpreadsheetAgent` for Excel/CSV analysis, and replaces static enum-based skill gating with a `SkillRegistry` that accepts runtime registration of custom skills.

**Tech Stack additions:**
- Phase 4: no new libraries
- Phase 5: `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `spacy`, `sentence-transformers`, `openpyxl`

---

## Phase 4

### Skill additions

| Skill | Tier | Agent | Notes |
|---|---|---|---|
| `HARD_RULE_ANOMALY` | ADVANCED | AnomalyAgent | Client-defined rule violations |
| `STATISTICAL_ANOMALY` | ADVANCED | AnomalyAgent | Baseline deviation detection |
| `FUNNEL_ANALYSIS` | ADVANCED | SQLAgent | Prompt template extension |
| `COHORT_ANALYSIS` | ADVANCED | SQLAgent | Prompt template extension |

### AnomalyAgent

**File:** `src/agents/anomaly_agent.py`

```python
class AnomalyAgent:
    def run(
        self,
        client_id: str,
        data: dict,           # rows + columns from SQLAgent
        rules: list[dict],    # client-defined hard rules (loaded from knowledge store)
        mode: str = "both",   # "hard", "statistical", or "both"
    ) -> AgentResult:
        ...
```

**Hard-rule detection:** Rules are stored as structured documents in the client's knowledge base (e.g. `{"metric": "churn_rate", "operator": ">", "threshold": 0.10, "severity": "critical"}`). The agent evaluates each rule against the current data and returns flagged violations with severity and context.

**Statistical detection:** Computes z-scores and IQR bounds from the last 30 periods of historical data (queried via SQLAgent). Flags any metric that deviates beyond 2.5 standard deviations as an anomaly with a description of the deviation magnitude and direction.

**Output:** `AgentResult` with `data={"anomalies": [{"metric": str, "value": float, "expected": float, "severity": str, "description": str}]}`.

**Orchestrator integration:** After the SQL step, if `HARD_RULE_ANOMALY` or `STATISTICAL_ANOMALY` is enabled and `sql_data` is present, the Orchestrator runs `AnomalyAgent.run()` and includes any anomalies in the response text and as a structured field on `Response`.

### InteractionMemoryLogger

**File:** `src/knowledge/interaction_memory.py`

```python
class InteractionMemoryLogger:
    def log(
        self,
        request: Request,
        response: Response,
        sql_queries: list[str] = [],
        corrections: list[str] = [],
    ) -> None:
        ...
```

Appends a JSON document to the client's ChromaDB collection under namespace `interaction_memory`. Each document contains: `request_text`, `response_text`, `sql_queries`, `corrections`, `timestamp`, `channel`. The `KnowledgeRetriever` already searches this collection — interaction history becomes available to future RAG context automatically.

The `Orchestrator` calls `InteractionMemoryLogger.log()` at the end of every successful `process()` call.

### FUNNEL_ANALYSIS and COHORT_ANALYSIS

These are prompt-template extensions of `SQLAgent`, not separate agents. When the plan includes `funnel=true` or `cohort=true`, the Orchestrator passes an enriched system prompt to `SQLAgent` that instructs it to write funnel or cohort SQL patterns.

- **Funnel:** ordered stages with conversion rates and drop-off counts between stages
- **Cohort:** users grouped by acquisition period, retention rates by period offset

The `_plan()` dict gains two new keys: `"funnel"` and `"cohort"`.

---

## Phase 5

### SkillRegistry

**File:** `src/core/skill_registry.py`

```python
class SkillDefinition(TypedDict):
    name: str
    description: str
    tier: Tier
    agent: str | None
    requires: list[str]

class SkillRegistry:
    _registry: dict[str, SkillDefinition] = {}

    def register(self, skill_id: str, defn: SkillDefinition) -> None: ...
    def get(self, skill_id: str) -> SkillDefinition | None: ...
    def all(self) -> dict[str, SkillDefinition]: ...
    def is_enabled(self, skill_id: str, config: ClientConfig) -> bool:
        return skill_id in config.enabled_skills

skill_registry = SkillRegistry()  # module-level singleton
```

All built-in skills are pre-registered in `skill_registry.py` at module load. `ClientConfig.enabled_skills` changes from `list[SkillModule]` to `list[str]`. The `SkillModule` enum stays as a namespace of string constants for type safety and IDE autocomplete — it is not the type constraint on `enabled_skills`.

Custom skills are registered at startup (e.g. from a JSON config or database) via `skill_registry.register(...)` with no code changes required.

**`SkillModule` additions for Phase 5:**

```python
class SkillModule(str, Enum):
    # ... existing ...
    NLP_MODELING = "nlp_modeling"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    SEGMENTATION = "segmentation"
    AB_TESTING = "ab_testing"
    DEEP_ANALYSIS = "deep_analysis"
    SPREADSHEET_ANALYSIS = "spreadsheet_analysis"
```

### AnalystAgent — ReAct loop

**File:** `src/agents/analyst_agent.py`

```python
class AnalystAgent:
    def __init__(self, llm, sql_agent, ml_agent, retriever): ...

    def run_deep(
        self,
        request: Request,
        config: ClientConfig,
        on_checkpoint: Callable[[int, str], None],
        max_steps: int = 10,
    ) -> AnalystResult: ...
```

**AnalystResult** (new model, extends AgentResult):

```python
class AnalystResult(AgentResult):
    steps: list[StepRecord] = []       # full trace of think→act→observe
    findings: list[str] = []
    solutions: list[dict] = []         # ranked, each has title/description/pros/cons
    recommendation: str = ""
```

**StepRecord:**

```python
class StepRecord(BaseModel):
    step: int
    thought: str
    tool: str
    tool_input: dict
    observation: str
```

**Loop mechanics:**

```
run_deep():
  history = []
  for step in range(1, max_steps + 1):
    thought, tool, tool_input = llm.think(request, history)
    if tool == "DONE": break
    observation = execute_tool(tool, tool_input, config)
    history.append(StepRecord(step, thought, tool, tool_input, observation))
    if step % 3 == 0:
      on_checkpoint(step, thought)   # posts progress to Slack/email
  return synthesize(request, history)
```

Hard cap: if `max_steps > 15`, it is clamped to 15.

**Tools available inside the loop:**

| Tool name | Action |
|---|---|
| `sql_query` | Natural-language SQL → rows via SQLAgent |
| `stat_test` | t-test / chi-square / ANOVA via `scipy.stats` |
| `cluster_segment` | KMeans segmentation via scikit-learn |
| `ml_prototype` | Train + evaluate best model via MLAgent |
| `knowledge_search` | RAG search via KnowledgeRetriever |

**Orchestrator integration:**

Two new steps prepend the existing pipeline when `DEEP_ANALYSIS` is in `config.enabled_skills`:

1. `detect_deep_intent(request)` — LLM classifies whether the request warrants a root-cause investigation (returns `True`/`False`)
2. If `True`: Orchestrator posts "Want me to do a deep dive on this? It may take a few minutes. Reply **yes** to proceed or **no** for a quick answer." and returns a `ClarificationState` with `is_resolved=False` and a `deep_dive_pending=True` flag
3. The next message in the same thread is evaluated: if it matches "yes"/"sure"/"go ahead" → delegates to `AnalystAgent.run_deep()`. Any other reply → existing single-pass pipeline
4. If intent is `False`: existing single-pass pipeline runs unchanged without asking

**`on_checkpoint` callback** posts a brief message to the originating channel every 3 steps: `"Step {n}: {thought}"`.

### MLAgent expansion

**File:** `src/agents/ml_agent.py` (extend existing)

**New mode: `build_model`**

```
build_model(data, task, target_col):
  task = "regression":
    candidates = [LinearRegression, Ridge, Lasso, ElasticNet,
                  RandomForestRegressor, GradientBoostingRegressor,
                  XGBRegressor, LGBMRegressor, SVR, KNeighborsRegressor,
                  MLPRegressor]
    → 5-fold cross-val, scorer=r2_score for each candidate
    → top 3 by R² → GridSearchCV on each
    → return best: model_name, best_params, feature_importance, metrics (R², RMSE, MAE)

  task = "classification":
    candidates = [LogisticRegression, RandomForestClassifier,
                  GradientBoostingClassifier, XGBClassifier, LGBMClassifier,
                  SVC, KNeighborsClassifier, GaussianNB, MLPClassifier]
    → 5-fold cross-val, scorer=f1_weighted for each candidate
    → top 3 by F1 → GridSearchCV on each
    → return best: model_name, best_params, feature_importance,
                   classification_report, confusion_matrix
```

**New mode: `nlp`** (activated by `NLP_MODELING` skill)

```
run_nlp(data, text_col, target_col=None, task="classify"):
  preprocess: spaCy tokenize + lemmatize + clean
  vectorize: TF-IDF (baseline) AND sentence-transformers embeddings (semantic)

  task = "classify":
    → run build_model() on both vector representations
    → return best pipeline (vectorizer + classifier) + metrics

  task = "ner":
    → spaCy NER → entity list with types and positions

  task = "similarity":
    → sentence-transformers cosine similarity matrix for input texts
```

All modes return `AgentResult` — no changes to Orchestrator or DeckAgent contracts.

### DeckAgent redesign

**File:** `src/agents/deck_agent.py` (redesign existing)

**Phase 1 — Storyline generation:**

```python
def build_storyline(
    self,
    analysis_text: str,
    findings: list[str],
    solutions: list[dict],
    recommendation: str,
) -> Storyline:
    ...
```

LLM prompt instructs it to return JSON with this structure:

```json
{
  "problem_statement": "str",
  "executive_summary": ["bullet 1", "bullet 2", "bullet 3"],
  "key_findings": [
    {
      "heading": "str",
      "body": "str",
      "so_what": "str",
      "chart_index": 0          // null if no chart; index into charts list
    }
  ],
  "solutions": [
    {
      "title": "str",
      "description": "str",
      "pros": ["str"],
      "cons": ["str"]
    }
  ],
  "recommended_solution": "str",
  "recommendation_rationale": "str",
  "conclusion": "str",
  "next_steps": ["str"]
}
```

`Storyline` is a Pydantic model matching this structure. `AgentResult` gains a `storyline: Storyline | None = None` field.

**Phase 2 — Slide generation (dynamic count):**

```python
def run(
    self,
    title: str,
    storyline: Storyline,
    charts: list[bytes] = [],
    template_path: str | None = None,
    max_solutions_inline: int = 3,
) -> AgentResult:
    ...
```

Slide sequence:

1. **Title slide** (fixed) — title, subtitle, date, client name. Full-bleed background.
2. **Executive summary** (fixed) — 3–5 bullet TL;DR + recommendation callout box in accent color.
3. **Problem statement** (fixed) — single crisp statement + supporting context bullets.
4. **Key findings** (dynamic) — one or two slides per finding depending on whether a chart is attached:
   - **Layout A** (no chart): heading + body left, "So what?" callout box right.
   - **Layout B** (with chart): heading top, chart left 60%, insight bullets right 40%. Chart matched via `chart_index` from storyline.
   - **Layout C** (data-heavy): heading top, styled table below with highlighted rows.
5. **Solutions** (dynamic) — if ≤ `max_solutions_inline`: one slide per solution with title, description, pros/cons columns. If > `max_solutions_inline`: single comparison table slide.
6. **Recommendation** (fixed) — recommended solution in large accent-colored callout, rationale bullets below.
7. **Conclusion + next steps** (fixed) — conclusion text + next steps with owner and timeline placeholders.
8. **Appendix** (optional) — any charts not matched to findings are placed here.

**Visual quality:**
- Client can supply a `.pptx` template file (`template_path`). Without one, a built-in default template with clean typography and color palette is used.
- "So what?" boxes, section dividers, and recommendation highlights are drawn as pptx shapes with theme colors.
- All text sizes, margins, and positions are defined as constants, not magic numbers.

### SpreadsheetAgent

**File:** `src/agents/spreadsheet_agent.py`

```python
class SpreadsheetAgent:
    def __init__(self, llm: LLMRouter): ...

    def run(
        self,
        file_bytes: bytes,
        filename: str,
        question: str,
        config: ClientConfig,
    ) -> AgentResult:
        ...
```

**Flow:**

1. Detect format from `filename` extension: `.csv` → `pandas.read_csv`, `.xlsx` / `.xls` → `pandas.read_excel` (via `openpyxl`)
2. LLM translates `question` into a sequence of pandas operations (filter, groupby, pivot, merge, describe)
3. Execute operations on the DataFrame
4. Return `AgentResult(data={"rows": ..., "columns": ..., "summary": ...})` — same contract as SQLAgent

**Orchestrator integration:** `Request` gains an optional `file_bytes: bytes | None = None` and `filename: str | None = None` field. If present and `SPREADSHEET_ANALYSIS` is in `enabled_skills`, Orchestrator routes to `SpreadsheetAgent` instead of `SQLAgent`. All downstream agents (ChartAgent, DeckAgent, AnalystAgent) work unchanged.

**New dependency:** `openpyxl>=3.1`

---

## Error handling

- **AnalystAgent loop**: if a tool raises an exception, the observation is `"Tool failed: {error}"` and the loop continues. After 3 consecutive tool failures, the loop exits early and `synthesize()` works with whatever history exists.
- **MLAgent model candidates**: if a candidate model fails to fit (e.g. SVR on large data), it is skipped silently and cross-val continues with remaining candidates. At least one model (LinearRegression / LogisticRegression) is always attempted as a fallback.
- **DeckAgent storyline**: if the LLM returns malformed JSON, `build_storyline()` falls back to a flat `Storyline` with `key_findings` derived from splitting `analysis_text` into paragraphs.
- **SpreadsheetAgent**: if pandas operations fail, the agent returns the raw `describe()` summary and logs the failed operation.

---

## Testing

- **AnomalyAgent**: unit tests with synthetic data for hard-rule violations and z-score detection. Mock SQLAgent.
- **InteractionMemoryLogger**: unit test that logs a request+response and verifies the document is retrievable via `KnowledgeRetriever`.
- **AnalystAgent**: unit test with mocked tools verifying loop terminates at `max_steps`, checkpoint fires every 3 steps, and `synthesize()` returns a valid `AnalystResult`.
- **MLAgent (build_model)**: integration test with a small synthetic dataset verifying best model is returned with non-null `feature_importance` and `metrics`.
- **MLAgent (nlp)**: integration test with 20 short labeled texts verifying classification pipeline runs end-to-end.
- **DeckAgent**: unit test verifying `build_storyline()` returns a valid `Storyline`, and `run()` produces a valid PPTX with correct slide count.
- **SpreadsheetAgent**: unit test with a synthetic in-memory CSV verifying filter + groupby operations produce expected output.
- **Integration (Phase 5)**: end-to-end test: Slack message triggers `detect_deep_intent` → mock user confirms → `AnalystAgent` runs 3 steps → checkpoint posted → `AnalystResult` returned.

---

## File structure

**New files:**
- `src/agents/anomaly_agent.py`
- `src/agents/analyst_agent.py`
- `src/agents/spreadsheet_agent.py`
- `src/knowledge/interaction_memory.py`
- `src/core/skill_registry.py`
- `tests/agents/test_anomaly_agent.py`
- `tests/agents/test_analyst_agent.py`
- `tests/agents/test_spreadsheet_agent.py`
- `tests/knowledge/test_interaction_memory.py`
- `tests/core/test_skill_registry.py`
- `tests/test_integration_phase4.py`
- `tests/test_integration_phase5.py`

**Modified files:**
- `src/core/models.py` — add `SkillModule` entries, `AnalystResult`, `StepRecord`, `Storyline`, `file_bytes`/`filename` on `Request`, `storyline` on `AgentResult`, `anomalies` on `Response`, `enabled_skills: list[str]`
- `src/core/skill_registry.py` — new, with pre-registered built-in skills
- `src/agents/ml_agent.py` — add `build_model` and `nlp` modes
- `src/agents/deck_agent.py` — redesign with `build_storyline` + dynamic slide generation
- `src/orchestrator/orchestrator.py` — add anomaly dispatch, memory logging, deep intent detection, spreadsheet routing, funnel/cohort plan keys
- `pyproject.toml` — add Phase 5 dependencies
- `main.py` — wire new agents
