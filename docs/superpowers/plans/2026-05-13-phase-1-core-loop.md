# AI Data Analyst — Phase 1: Core Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working AI Data Analyst that can be @mentioned in Slack, asks clarifying questions when needed, queries a SQL database, and replies with analysis + chart.

**Architecture:** Modular Orchestrator pattern — a central Orchestrator delegates to a SQL Agent and Chart Agent based on the client's enabled skill modules. A Clarification Checker handles ambiguous requests before execution. A Slack Channel adapter normalizes incoming events and sends replies.

**Tech Stack:** Python 3.11+, Anthropic SDK (Claude), OpenAI SDK (GPT-4o), ChromaDB (vector DB), SQLAlchemy (SQL), Matplotlib (charts), Slack Bolt (Slack integration), FastAPI (HTTP server), Pytest (testing), Pydantic v2 (models)

---

## File Map

```
ai_talent/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # App & client configuration via pydantic-settings
│   │   ├── models.py           # Pydantic models: Request, Response, ClientConfig, etc.
│   │   └── llm.py              # LLMRouter: multi-provider abstraction (Claude / GPT-4o / Haiku)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # ChromaDB wrapper with per-client namespace isolation
│   │   ├── ingestion.py        # Document parser + chunker + embedder + store
│   │   └── retriever.py        # Semantic search over client namespace
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── sql.py              # SQLAlchemy read-only connector + schema introspection
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sql_agent.py        # Translates request → SQL → executes → returns DataFrame
│   │   └── chart_agent.py      # Data → Matplotlib chart → PNG bytes
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── clarifier.py        # Clarity check + clarification loop (max 2 rounds)
│   │   └── orchestrator.py     # Full flow: understand → validate → plan → delegate → respond
│   ├── channels/
│   │   ├── __init__.py
│   │   └── slack.py            # Slack Bolt app: receive @mentions, send text + chart replies
│   └── skill_modules.py        # SkillModuleRegistry: per-client enabled skills + tier defaults
├── tests/
│   ├── core/
│   │   ├── test_models.py
│   │   └── test_llm.py
│   ├── knowledge/
│   │   ├── test_vector_store.py
│   │   ├── test_ingestion.py
│   │   └── test_retriever.py
│   ├── connectors/
│   │   └── test_sql.py
│   ├── agents/
│   │   ├── test_sql_agent.py
│   │   └── test_chart_agent.py
│   ├── orchestrator/
│   │   ├── test_clarifier.py
│   │   └── test_orchestrator.py
│   └── channels/
│       └── test_slack.py
├── .env.example
├── pyproject.toml
└── main.py                     # FastAPI entry point
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `main.py`
- Create: all `__init__.py` files

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ai-talent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.30.0",
    "openai>=1.30.0",
    "chromadb>=0.5.0",
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "slack-bolt>=1.18",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pdfplumber>=0.10",
    "python-docx>=1.0",
    "openpyxl>=3.1",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create .env.example**

```bash
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_SIGNING_SECRET=your-signing-secret
CHROMADB_PERSIST_DIR=.chromadb
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p src/core src/knowledge src/connectors src/agents src/orchestrator src/channels
mkdir -p tests/core tests/knowledge tests/connectors tests/agents tests/orchestrator tests/channels
touch src/__init__.py src/core/__init__.py src/knowledge/__init__.py
touch src/connectors/__init__.py src/agents/__init__.py src/orchestrator/__init__.py src/channels/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/knowledge/__init__.py
touch tests/connectors/__init__.py tests/agents/__init__.py tests/orchestrator/__init__.py tests/channels/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: All packages install without errors.

- [ ] **Step 5: Verify pytest works**

```bash
pytest --collect-only
```

Expected: `0 tests collected` (no tests yet — that's fine).

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml .env.example src/ tests/
git commit -m "feat: project scaffold and dependencies"
```

---

## Task 2: Core Models

**Files:**
- Create: `src/core/models.py`
- Create: `tests/core/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_models.py
from src.core.models import (
    Channel, SkillModule, Tier, Request, ClientConfig,
    ClarificationState, AgentResult, Response
)
from datetime import datetime

def test_request_model():
    r = Request(
        channel=Channel.SLACK,
        sender_id="U123",
        sender_name="Budi",
        text="show me churn rate",
        thread_id="1234567890.000100",
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-abc",
    )
    assert r.channel == Channel.SLACK
    assert r.client_id == "client-abc"

def test_client_config_basic_tier():
    config = ClientConfig(
        client_id="client-abc",
        name="Test Corp",
        tier=Tier.BASIC,
        enabled_skills=[
            SkillModule.SQL_QUERYING,
            SkillModule.DATA_VISUALIZATION,
            SkillModule.REPORT_GENERATION,
            SkillModule.SCHEDULED_REPORTING,
            SkillModule.HARD_RULE_ANOMALY,
        ],
        account_mode="vendor",
        active_channels=[Channel.SLACK],
    )
    assert SkillModule.SQL_QUERYING in config.enabled_skills
    assert SkillModule.MACHINE_LEARNING not in config.enabled_skills

def test_clarification_state_defaults():
    r = Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="A",
        text="show data", timestamp="2026-01-01T00:00:00", client_id="c1"
    )
    state = ClarificationState(original_request=r)
    assert state.rounds == 0
    assert state.is_resolved is False
    assert state.assumptions == []

def test_agent_result_success():
    result = AgentResult(
        agent_name="sql_agent",
        success=True,
        data={"query": "SELECT 1", "rows": [{"count": 1}], "columns": ["count"]}
    )
    assert result.success is True
    assert result.error is None

def test_response_defaults():
    resp = Response(request_id="req-1", text="Here is your analysis.")
    assert resp.charts == []
    assert resp.assumptions == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_models.py -v
```

Expected: `ImportError` — models not defined yet.

- [ ] **Step 3: Write implementation**

```python
# src/core/models.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Channel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    JIRA = "jira"


class SkillModule(str, Enum):
    SQL_QUERYING = "sql_querying"
    DATA_VISUALIZATION = "data_visualization"
    REPORT_GENERATION = "report_generation"
    SCHEDULED_REPORTING = "scheduled_reporting"
    HARD_RULE_ANOMALY = "hard_rule_anomaly"
    SPREADSHEET_ANALYSIS = "spreadsheet_analysis"
    LOOKER_INTEGRATION = "looker_integration"
    STATISTICAL_ANOMALY = "statistical_anomaly"
    MACHINE_LEARNING = "machine_learning"
    PRESENTATION_BUILDING = "presentation_building"


class Tier(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"


class Request(BaseModel):
    channel: Channel
    sender_id: str
    sender_name: str
    text: str
    thread_id: Optional[str] = None
    timestamp: str
    client_id: str


class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[SkillModule]
    account_mode: str  # "vendor" or "dedicated"
    active_channels: list[Channel]


class ClarificationState(BaseModel):
    original_request: Request
    rounds: int = 0
    questions_asked: list[str] = []
    answers_received: list[str] = []
    is_resolved: bool = False
    assumptions: list[str] = []


class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: Optional[dict] = None
    chart_png: Optional[bytes] = None
    error: Optional[str] = None


class Response(BaseModel):
    request_id: str
    text: str
    charts: list[bytes] = []
    assumptions: list[str] = []
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_models.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/core/models.py tests/core/test_models.py
git commit -m "feat: core models (Request, ClientConfig, ClarificationState, AgentResult, Response)"
```

---

## Task 3: LLM Abstraction Layer

**Files:**
- Create: `src/core/llm.py`
- Create: `tests/core/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_llm.py
from unittest.mock import MagicMock, patch
from src.core.llm import LLMRouter
from src.core.models import TaskType


def make_router():
    return LLMRouter(anthropic_api_key="fake-anthropic", openai_api_key="fake-openai")


def test_reasoning_routes_to_claude(mocker):
    router = make_router()
    mock_complete = mocker.patch.object(router, "_claude", return_value="claude reply")
    result = router.complete(TaskType.REASONING, "sys", "user")
    mock_complete.assert_called_once_with("sys", "user", model="claude-sonnet-4-6")
    assert result == "claude reply"


def test_tool_routes_to_gpt(mocker):
    router = make_router()
    mock_complete = mocker.patch.object(router, "_gpt", return_value="gpt reply")
    result = router.complete(TaskType.TOOL, "sys", "user")
    mock_complete.assert_called_once_with("sys", "user", model="gpt-4o")
    assert result == "gpt reply"


def test_simple_routes_to_haiku(mocker):
    router = make_router()
    mock_complete = mocker.patch.object(router, "_claude", return_value="haiku reply")
    result = router.complete(TaskType.SIMPLE, "sys", "user")
    mock_complete.assert_called_once_with("sys", "user", model="claude-haiku-4-5-20251001")
    assert result == "haiku reply"
```

- [ ] **Step 2: Add TaskType to models.py**

```python
# Add to src/core/models.py after the existing enums:

class TaskType(str, Enum):
    REASONING = "reasoning"   # → Claude Sonnet (complex analysis)
    TOOL = "tool"             # → GPT-4o (tool-heavy tasks like SQL generation)
    SIMPLE = "simple"         # → Claude Haiku (cheap, fast: clarification checks)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/core/test_llm.py -v
```

Expected: `ImportError` — LLMRouter not defined yet.

- [ ] **Step 4: Write implementation**

```python
# src/core/llm.py
from anthropic import Anthropic
from openai import OpenAI
from src.core.models import TaskType


class LLMRouter:
    def __init__(self, anthropic_api_key: str, openai_api_key: str):
        self._anthropic = Anthropic(api_key=anthropic_api_key)
        self._openai = OpenAI(api_key=openai_api_key)

    def complete(self, task_type: TaskType, system: str, user: str) -> str:
        if task_type == TaskType.REASONING:
            return self._claude(system, user, model="claude-sonnet-4-6")
        elif task_type == TaskType.TOOL:
            return self._gpt(system, user, model="gpt-4o")
        else:
            return self._claude(system, user, model="claude-haiku-4-5-20251001")

    def _claude(self, system: str, user: str, model: str) -> str:
        msg = self._anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def _gpt(self, system: str, user: str, model: str) -> str:
        resp = self._openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/core/test_llm.py -v
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/core/llm.py src/core/models.py tests/core/test_llm.py
git commit -m "feat: LLM abstraction layer with Claude/GPT-4o/Haiku routing"
```

---

## Task 4: Skill Module Registry

**Files:**
- Create: `src/skill_modules.py`
- Create: `tests/test_skill_modules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skill_modules.py
from src.skill_modules import SkillModuleRegistry
from src.core.models import SkillModule, Tier


def test_basic_tier_has_core_skills():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.BASIC)
    assert SkillModule.SQL_QUERYING in skills
    assert SkillModule.DATA_VISUALIZATION in skills
    assert SkillModule.REPORT_GENERATION in skills
    assert SkillModule.SCHEDULED_REPORTING in skills
    assert SkillModule.HARD_RULE_ANOMALY in skills


def test_basic_tier_excludes_advanced_skills():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.BASIC)
    assert SkillModule.MACHINE_LEARNING not in skills
    assert SkillModule.PRESENTATION_BUILDING not in skills
    assert SkillModule.STATISTICAL_ANOMALY not in skills


def test_advanced_tier_includes_sheets_and_looker():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED)
    assert SkillModule.SPREADSHEET_ANALYSIS in skills
    assert SkillModule.LOOKER_INTEGRATION in skills
    assert SkillModule.STATISTICAL_ANOMALY in skills


def test_enterprise_tier_includes_ml_and_deck():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.ENTERPRISE)
    assert SkillModule.MACHINE_LEARNING in skills
    assert SkillModule.PRESENTATION_BUILDING in skills


def test_is_enabled():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.BASIC)
    assert SkillModuleRegistry.is_enabled(SkillModule.SQL_QUERYING, skills) is True
    assert SkillModuleRegistry.is_enabled(SkillModule.MACHINE_LEARNING, skills) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_skill_modules.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/skill_modules.py
from src.core.models import SkillModule, Tier

_BASIC = [
    SkillModule.SQL_QUERYING,
    SkillModule.DATA_VISUALIZATION,
    SkillModule.REPORT_GENERATION,
    SkillModule.SCHEDULED_REPORTING,
    SkillModule.HARD_RULE_ANOMALY,
]

_ADVANCED = _BASIC + [
    SkillModule.SPREADSHEET_ANALYSIS,
    SkillModule.LOOKER_INTEGRATION,
    SkillModule.STATISTICAL_ANOMALY,
]

_ENTERPRISE = _ADVANCED + [
    SkillModule.MACHINE_LEARNING,
    SkillModule.PRESENTATION_BUILDING,
]

_TIER_MAP = {
    Tier.BASIC: _BASIC,
    Tier.ADVANCED: _ADVANCED,
    Tier.ENTERPRISE: _ENTERPRISE,
}


class SkillModuleRegistry:
    @staticmethod
    def defaults_for_tier(tier: Tier) -> list[SkillModule]:
        return list(_TIER_MAP[tier])

    @staticmethod
    def is_enabled(skill: SkillModule, enabled: list[SkillModule]) -> bool:
        return skill in enabled
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_skill_modules.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/skill_modules.py tests/test_skill_modules.py
git commit -m "feat: skill module registry with tier-based defaults"
```

---

## Task 5: Vector Store

**Files:**
- Create: `src/knowledge/vector_store.py`
- Create: `tests/knowledge/test_vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/knowledge/test_vector_store.py
import pytest
import tempfile
from src.knowledge.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path), openai_api_key="fake-key")


def test_add_and_search(store, mocker):
    mocker.patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction.__call__",
        return_value=[[0.1, 0.2, 0.3]]
    )
    store.add(
        client_id="client-1",
        texts=["NPL means non-performing loans overdue by 90+ days"],
        ids=["doc-1"],
    )
    results = store.search("client-1", "what is NPL", top_k=1)
    assert len(results) == 1
    assert "NPL" in results[0]


def test_client_isolation(store, mocker):
    mocker.patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction.__call__",
        return_value=[[0.1, 0.2, 0.3]]
    )
    store.add("client-a", ["Client A secret data"], ["id-a"])
    store.add("client-b", ["Client B secret data"], ["id-b"])

    results_a = store.search("client-a", "secret data", top_k=5)
    assert all("Client A" in r for r in results_a)

    results_b = store.search("client-b", "secret data", top_k=5)
    assert all("Client B" in r for r in results_b)


def test_delete_client(store, mocker):
    mocker.patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction.__call__",
        return_value=[[0.1, 0.2, 0.3]]
    )
    store.add("client-del", ["some data"], ["id-1"])
    store.delete_client("client-del")
    results = store.search("client-del", "some data", top_k=1)
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/knowledge/test_vector_store.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/knowledge/vector_store.py
import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    def __init__(self, persist_dir: str, openai_api_key: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small",
        )

    def _collection(self, client_id: str):
        return self._client.get_or_create_collection(
            name=f"client_{client_id}",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        client_id: str,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        col = self._collection(client_id)
        col.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas or [{} for _ in texts],
        )

    def search(self, client_id: str, query: str, top_k: int = 5) -> list[str]:
        try:
            col = self._collection(client_id)
            if col.count() == 0:
                return []
            results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    def delete_client(self, client_id: str) -> None:
        try:
            self._client.delete_collection(f"client_{client_id}")
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/knowledge/test_vector_store.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/vector_store.py tests/knowledge/test_vector_store.py
git commit -m "feat: ChromaDB vector store with per-client namespace isolation"
```

---

## Task 6: Document Ingestion Pipeline

**Files:**
- Create: `src/knowledge/ingestion.py`
- Create: `tests/knowledge/test_ingestion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/knowledge/test_ingestion.py
import os
import tempfile
import pytest
from unittest.mock import MagicMock
from src.knowledge.ingestion import DocumentIngester


@pytest.fixture
def ingester():
    store = MagicMock()
    return DocumentIngester(vector_store=store)


def test_ingest_csv(ingester, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,value\nalpha,1\nbeta,2\n")
    count = ingester.ingest_file("client-1", str(csv_file))
    assert count >= 1
    ingester._store.add.assert_called_once()
    args = ingester._store.add.call_args
    assert args[0][0] == "client-1"


def test_ingest_txt(ingester, tmp_path):
    txt_file = tmp_path / "rules.txt"
    txt_file.write_text("NPL means non-performing loans. " * 50)
    count = ingester.ingest_file("client-1", str(txt_file))
    assert count >= 1


def test_unsupported_format_raises(ingester, tmp_path):
    bad_file = tmp_path / "file.json"
    bad_file.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        ingester.ingest_file("client-1", str(bad_file))


def test_chunking_splits_large_text(ingester):
    text = "word " * 2000
    chunks = ingester._chunk(text)
    assert len(chunks) > 1
    assert all(len(c.split()) <= ingester._chunk_size for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/knowledge/test_ingestion.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/knowledge/ingestion.py
import uuid
from pathlib import Path

import pandas as pd

from src.knowledge.vector_store import VectorStore


class DocumentIngester:
    def __init__(self, vector_store: VectorStore, chunk_size: int = 500):
        self._store = vector_store
        self._chunk_size = chunk_size

    def ingest_file(self, client_id: str, file_path: str) -> int:
        path = Path(file_path)
        suffix = path.suffix.lower()

        parsers = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".xlsx": self._parse_xlsx,
            ".xls": self._parse_xlsx,
            ".csv": self._parse_csv,
            ".txt": lambda p: Path(p).read_text(encoding="utf-8", errors="ignore"),
        }

        if suffix not in parsers:
            raise ValueError(f"Unsupported file type: {suffix}")

        text = parsers[suffix](file_path)
        chunks = self._chunk(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": path.name, "client_id": client_id} for _ in chunks]
        self._store.add(client_id, chunks, ids, metadatas)
        return len(chunks)

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self._chunk_size):
            chunk = " ".join(words[i : i + self._chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks or [text[:200]]

    def _parse_pdf(self, path: str) -> str:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def _parse_docx(self, path: str) -> str:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    def _parse_xlsx(self, path: str) -> str:
        sheets = pd.read_excel(path, sheet_name=None)
        parts = [f"Sheet: {name}\n{df.to_string()}" for name, df in sheets.items()]
        return "\n\n".join(parts)

    def _parse_csv(self, path: str) -> str:
        return pd.read_csv(path).to_string()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/knowledge/test_ingestion.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/ingestion.py tests/knowledge/test_ingestion.py
git commit -m "feat: document ingestion pipeline (PDF, DOCX, XLSX, CSV, TXT)"
```

---

## Task 7: Knowledge Retriever

**Files:**
- Create: `src/knowledge/retriever.py`
- Create: `tests/knowledge/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/knowledge/test_retriever.py
from unittest.mock import MagicMock
from src.knowledge.retriever import KnowledgeRetriever


def test_search_delegates_to_store():
    store = MagicMock()
    store.search.return_value = ["NPL means 90+ days overdue", "Credit risk defined as..."]
    retriever = KnowledgeRetriever(store=store, top_k=5)

    results = retriever.search("client-1", "what is NPL")

    store.search.assert_called_once_with("client-1", "what is NPL", 5)
    assert len(results) == 2


def test_search_returns_empty_when_no_knowledge():
    store = MagicMock()
    store.search.return_value = []
    retriever = KnowledgeRetriever(store=store)

    results = retriever.search("client-new", "anything")
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/knowledge/test_retriever.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/knowledge/retriever.py
from src.knowledge.vector_store import VectorStore


class KnowledgeRetriever:
    def __init__(self, store: VectorStore, top_k: int = 5):
        self._store = store
        self._top_k = top_k

    def search(self, client_id: str, query: str) -> list[str]:
        return self._store.search(client_id, query, self._top_k)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/knowledge/test_retriever.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/retriever.py tests/knowledge/test_retriever.py
git commit -m "feat: knowledge retriever for semantic search over client namespace"
```

---

## Task 8: SQL Connector

**Files:**
- Create: `src/connectors/sql.py`
- Create: `tests/connectors/test_sql.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/connectors/test_sql.py
import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from src.connectors.sql import SQLConnector


@pytest.fixture
def connector():
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE sales (id INTEGER, amount REAL, region TEXT)"))
        conn.execute(text("INSERT INTO sales VALUES (1, 1000.0, 'Jakarta')"))
        conn.execute(text("INSERT INTO sales VALUES (2, 2000.0, 'Surabaya')"))
        conn.commit()
    return SQLConnector(connection_url="sqlite:///:memory:", engine=engine)


def test_execute_select(connector):
    df = connector.execute("SELECT * FROM sales")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "amount" in df.columns


def test_get_schema(connector):
    schema = connector.get_schema()
    assert "sales" in schema
    col_names = [c["name"] for c in schema["sales"]]
    assert "id" in col_names
    assert "amount" in col_names


def test_blocks_insert(connector):
    with pytest.raises(ValueError, match="Read-only"):
        connector.execute("INSERT INTO sales VALUES (3, 500.0, 'Bandung')")


def test_blocks_drop(connector):
    with pytest.raises(ValueError, match="Read-only"):
        connector.execute("DROP TABLE sales")


def test_blocks_update(connector):
    with pytest.raises(ValueError, match="Read-only"):
        connector.execute("UPDATE sales SET amount = 0 WHERE id = 1")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/connectors/test_sql.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/connectors/sql.py
from sqlalchemy import create_engine, text, inspect, Engine
import pandas as pd


_WRITE_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE")


class SQLConnector:
    def __init__(self, connection_url: str, engine: Engine | None = None):
        self._engine = engine or create_engine(connection_url)

    def get_schema(self) -> dict:
        inspector = inspect(self._engine)
        return {
            table: [
                {"name": col["name"], "type": str(col["type"])}
                for col in inspector.get_columns(table)
            ]
            for table in inspector.get_table_names()
        }

    def execute(self, query: str) -> pd.DataFrame:
        first_word = query.strip().split()[0].upper()
        if first_word in _WRITE_KEYWORDS:
            raise ValueError(f"Read-only access only. '{first_word}' is not permitted.")
        with self._engine.connect() as conn:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/connectors/test_sql.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/connectors/sql.py tests/connectors/test_sql.py
git commit -m "feat: read-only SQL connector with schema introspection"
```

---

## Task 9: SQL Agent

**Files:**
- Create: `src/agents/sql_agent.py`
- Create: `tests/agents/test_sql_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_sql_agent.py
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.agents.sql_agent import SQLAgent
from src.core.models import TaskType


@pytest.fixture
def sql_agent():
    llm = MagicMock()
    connector = MagicMock()
    store = MagicMock()
    return SQLAgent(llm=llm, connector=connector, store=store), llm, connector, store


def test_successful_query(sql_agent):
    agent, llm, connector, store = sql_agent
    llm.complete.return_value = "SELECT region, SUM(amount) FROM sales GROUP BY region"
    connector.get_schema.return_value = {"sales": [{"name": "region", "type": "TEXT"}, {"name": "amount", "type": "REAL"}]}
    connector.execute.return_value = pd.DataFrame([{"region": "Jakarta", "SUM(amount)": 1000.0}])

    result = agent.run(
        client_id="client-1",
        request="show total sales by region",
        context=["Sales data is in the sales table"],
    )

    assert result.success is True
    assert result.data["query"] == "SELECT region, SUM(amount) FROM sales GROUP BY region"
    assert len(result.data["rows"]) == 1
    store.add.assert_called_once()


def test_failed_query_returns_error(sql_agent):
    agent, llm, connector, store = sql_agent
    llm.complete.return_value = "SELECT * FROM nonexistent"
    connector.get_schema.return_value = {}
    connector.execute.side_effect = Exception("table not found")

    result = agent.run("client-1", "bad request", [])

    assert result.success is False
    assert "table not found" in result.error
    store.add.assert_not_called()


def test_uses_tool_task_type(sql_agent):
    agent, llm, connector, store = sql_agent
    llm.complete.return_value = "SELECT 1"
    connector.get_schema.return_value = {}
    connector.execute.return_value = pd.DataFrame([{"1": 1}])

    agent.run("client-1", "test", [])

    call_args = llm.complete.call_args
    assert call_args[0][0] == TaskType.TOOL
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_sql_agent.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/agents/sql_agent.py
import json
import uuid

from src.core.llm import LLMRouter
from src.core.models import TaskType, AgentResult
from src.connectors.sql import SQLConnector
from src.knowledge.vector_store import VectorStore


class SQLAgent:
    def __init__(self, llm: LLMRouter, connector: SQLConnector, store: VectorStore):
        self._llm = llm
        self._connector = connector
        self._store = store

    def run(self, client_id: str, request: str, context: list[str]) -> AgentResult:
        schema = self._connector.get_schema()
        system = (
            "You are a SQL expert. Given a user request, database schema, and business context, "
            "generate a single valid read-only SQL SELECT query. "
            "Return ONLY the SQL query — no explanation, no markdown, no backticks."
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

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_sql_agent.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/agents/sql_agent.py tests/agents/test_sql_agent.py
git commit -m "feat: SQL agent — request to SQL to results with knowledge base logging"
```

---

## Task 10: Chart Agent

**Files:**
- Create: `src/agents/chart_agent.py`
- Create: `tests/agents/test_chart_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_chart_agent.py
from src.agents.chart_agent import ChartAgent


def make_agent():
    return ChartAgent()


def test_single_value_skips_chart():
    agent = make_agent()
    data = {"rows": [{"count": 42}], "columns": ["count"]}
    result = agent.run(data)
    assert result.success is True
    assert result.data == {"skipped": True}
    assert result.chart_png is None


def test_categorical_data_produces_png():
    agent = make_agent()
    data = {
        "rows": [
            {"region": "Jakarta", "total": 1000},
            {"region": "Surabaya", "total": 2000},
            {"region": "Bandung", "total": 500},
        ],
        "columns": ["region", "total"],
    }
    result = agent.run(data)
    assert result.success is True
    assert isinstance(result.chart_png, bytes)
    assert result.chart_png[:4] == b"\x89PNG"


def test_empty_data_returns_error():
    agent = make_agent()
    result = agent.run({"rows": [], "columns": ["x", "y"]})
    assert result.success is False
    assert result.error is not None


def test_missing_data_returns_error():
    agent = make_agent()
    result = agent.run({})
    assert result.success is False


def test_time_series_data_produces_png():
    agent = make_agent()
    data = {
        "rows": [{"month": i, "revenue": i * 100} for i in range(1, 7)],
        "columns": ["month", "revenue"],
    }
    result = agent.run(data)
    assert result.success is True
    assert isinstance(result.chart_png, bytes)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_chart_agent.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/agents/chart_agent.py
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.core.models import AgentResult


class ChartAgent:
    def run(self, data: dict) -> AgentResult:
        if not data.get("rows") or not data.get("columns"):
            return AgentResult(agent_name="chart_agent", success=False, error="No data to chart")

        df = pd.DataFrame(data["rows"], columns=data["columns"])

        if df.empty:
            return AgentResult(agent_name="chart_agent", success=False, error="Empty dataset")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        # Single value — no chart needed
        if len(df) == 1 and len(numeric_cols) <= 1 and not non_numeric_cols:
            return AgentResult(agent_name="chart_agent", success=True, data={"skipped": True})

        png = self._render(df, numeric_cols, non_numeric_cols)
        if png is None:
            return AgentResult(agent_name="chart_agent", success=True, data={"skipped": True})

        return AgentResult(agent_name="chart_agent", success=True, chart_png=png)

    def _render(self, df: pd.DataFrame, numeric_cols: list, non_numeric_cols: list) -> bytes | None:
        fig, ax = plt.subplots(figsize=(10, 6))

        if non_numeric_cols and numeric_cols:
            x_col, y_col = non_numeric_cols[0], numeric_cols[0]
            ax.bar(df[x_col].astype(str), df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{y_col} by {x_col}")
        elif len(numeric_cols) >= 2:
            ax.plot(df[numeric_cols[0]], df[numeric_cols[1]], marker="o")
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"{numeric_cols[1]} over {numeric_cols[0]}")
        else:
            plt.close(fig)
            return None

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_chart_agent.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/agents/chart_agent.py tests/agents/test_chart_agent.py
git commit -m "feat: chart agent — auto-selects bar or line chart from data shape, returns PNG"
```

---

## Task 11: Clarification Checker

**Files:**
- Create: `src/orchestrator/clarifier.py`
- Create: `tests/orchestrator/test_clarifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_clarifier.py
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from src.orchestrator.clarifier import ClarificationChecker
from src.core.models import Request, Channel, ClarificationState


def make_request(text: str = "show me data") -> Request:
    return Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Budi",
        text=text,
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-1",
    )


@pytest.fixture
def checker():
    llm = MagicMock()
    return ClarificationChecker(llm=llm), llm


def test_clear_request_resolves_immediately(checker):
    c, llm = checker
    llm.complete.return_value = '{"is_clear": true, "questions": []}'
    state = c.check(make_request("show total sales by region for last month"), [])
    assert state.is_resolved is True
    assert state.questions_asked == []


def test_ambiguous_request_generates_questions(checker):
    c, llm = checker
    llm.complete.return_value = '{"is_clear": false, "questions": ["Which time period?", "Which region?"]}'
    state = c.check(make_request("show me the sales data"), [])
    assert state.is_resolved is False
    assert len(state.questions_asked) == 2
    assert state.rounds == 1


def test_exceeding_max_rounds_forces_resolution(checker):
    c, llm = checker
    llm.complete.return_value = '["Assuming last month", "Assuming all regions"]'
    request = make_request("show data")
    existing_state = ClarificationState(
        original_request=request,
        rounds=2,
        questions_asked=["Which period?", "Which region?"],
        answers_received=[],
    )
    state = c.check(request, [], existing_state)
    assert state.is_resolved is True
    assert len(state.assumptions) >= 1


def test_invalid_llm_json_defaults_to_clear(checker):
    c, llm = checker
    llm.complete.return_value = "not valid json"
    state = c.check(make_request("some request"), [])
    assert state.is_resolved is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/orchestrator/test_clarifier.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/orchestrator/clarifier.py
import json

from src.core.llm import LLMRouter
from src.core.models import TaskType, Request, ClarificationState

MAX_ROUNDS = 2


class ClarificationChecker:
    def __init__(self, llm: LLMRouter):
        self._llm = llm

    def check(
        self,
        request: Request,
        context: list[str],
        state: ClarificationState | None = None,
    ) -> ClarificationState:
        if state is None:
            state = ClarificationState(original_request=request)

        if state.rounds >= MAX_ROUNDS:
            assumptions = self._generate_assumptions(request, state, context)
            return ClarificationState(
                original_request=request,
                rounds=state.rounds,
                questions_asked=state.questions_asked,
                answers_received=state.answers_received,
                is_resolved=True,
                assumptions=assumptions,
            )

        system = (
            "You are an AI data analyst assessing if a request is specific enough to execute. "
            "A request needs: what metric/data, what time period (if relevant), what filters (if relevant). "
            'Return JSON only: {"is_clear": true/false, "questions": ["q1"] | []}. '
            "If clear, questions must be empty. If not, provide 1-2 specific questions only."
        )
        user = (
            f"Request: {request.text}\n\n"
            f"Business context:\n{chr(10).join(context)}\n\n"
            "Is this specific enough to execute?"
        )

        raw = self._llm.complete(TaskType.SIMPLE, system, user)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            state.is_resolved = True
            return state

        if result.get("is_clear"):
            state.is_resolved = True
        else:
            state.questions_asked.extend(result.get("questions", []))
            state.rounds += 1

        return state

    def _generate_assumptions(
        self, request: Request, state: ClarificationState, context: list[str]
    ) -> list[str]:
        system = (
            "You are an AI data analyst. State the assumptions you will use to proceed with an ambiguous request. "
            "Return a JSON list of strings."
        )
        user = (
            f"Request: {request.text}\n"
            f"Questions asked: {state.questions_asked}\n"
            f"Answers received: {state.answers_received}\n"
            f"Context: {chr(10).join(context)}"
        )
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ["Proceeding with best interpretation of the original request."]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/orchestrator/test_clarifier.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/clarifier.py tests/orchestrator/test_clarifier.py
git commit -m "feat: clarification checker with max-2-round loop and assumption fallback"
```

---

## Task 12: Orchestrator

**Files:**
- Create: `src/orchestrator/orchestrator.py`
- Create: `tests/orchestrator/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestrator/test_orchestrator.py
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from src.orchestrator.orchestrator import Orchestrator
from src.core.models import (
    Request, Channel, ClientConfig, Tier, SkillModule,
    ClarificationState, AgentResult, Response
)


def make_request(text: str = "show total sales by region") -> Request:
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Budi",
        text=text, timestamp=datetime.utcnow().isoformat(), client_id="client-1",
    )


def make_config(tier: Tier = Tier.ADVANCED) -> ClientConfig:
    from src.skill_modules import SkillModuleRegistry
    return ClientConfig(
        client_id="client-1", name="Test Corp", tier=tier,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(tier),
        account_mode="vendor", active_channels=[Channel.SLACK],
    )


@pytest.fixture
def orchestrator():
    llm = MagicMock()
    retriever = MagicMock()
    clarifier = MagicMock()
    sql_agent = MagicMock()
    chart_agent = MagicMock()
    return Orchestrator(llm, retriever, clarifier, sql_agent, chart_agent), \
           llm, retriever, clarifier, sql_agent, chart_agent


def test_returns_clarification_state_when_unclear(orchestrator):
    orch, llm, retriever, clarifier, sql_agent, chart_agent = orchestrator
    retriever.search.return_value = []
    clarifier.check.return_value = ClarificationState(
        original_request=make_request(), rounds=1,
        questions_asked=["Which time period?"], is_resolved=False,
    )

    result = orch.process(make_request("show data"), make_config())
    assert isinstance(result, ClarificationState)
    sql_agent.run.assert_not_called()


def test_returns_response_when_clear(orchestrator):
    orch, llm, retriever, clarifier, sql_agent, chart_agent = orchestrator
    retriever.search.return_value = ["Sales table contains region and amount"]
    clarifier.check.return_value = ClarificationState(
        original_request=make_request(), is_resolved=True,
    )
    llm.complete.side_effect = [
        '{"sql": true, "chart": true}',   # plan
        "Here is the sales analysis...",   # response text
    ]
    sql_agent.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT 1", "rows": [{"region": "Jakarta", "total": 1000}], "columns": ["region", "total"]}
    )
    chart_agent.run.return_value = AgentResult(
        agent_name="chart_agent", success=True, chart_png=b"\x89PNG..."
    )

    result = orch.process(make_request(), make_config())
    assert isinstance(result, Response)
    assert result.text == "Here is the sales analysis..."
    assert len(result.charts) == 1


def test_sql_skill_disabled_skips_sql_agent(orchestrator):
    orch, llm, retriever, clarifier, sql_agent, chart_agent = orchestrator
    retriever.search.return_value = []
    clarifier.check.return_value = ClarificationState(
        original_request=make_request(), is_resolved=True,
    )
    llm.complete.side_effect = [
        '{"sql": true, "chart": true}',
        "I cannot run SQL queries for your current plan.",
    ]
    config = ClientConfig(
        client_id="client-1", name="Test", tier=Tier.BASIC,
        enabled_skills=[SkillModule.REPORT_GENERATION],  # no SQL skill
        account_mode="vendor", active_channels=[Channel.SLACK],
    )

    result = orch.process(make_request(), config)
    assert isinstance(result, Response)
    sql_agent.run.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/orchestrator/test_orchestrator.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/orchestrator/orchestrator.py
import json
import uuid

from src.core.llm import LLMRouter
from src.core.models import (
    TaskType, Request, Response, ClientConfig, SkillModule, ClarificationState, AgentResult
)
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker


class Orchestrator:
    def __init__(
        self,
        llm: LLMRouter,
        retriever: KnowledgeRetriever,
        clarifier: ClarificationChecker,
        sql_agent: SQLAgent,
        chart_agent: ChartAgent,
    ):
        self._llm = llm
        self._retriever = retriever
        self._clarifier = clarifier
        self._sql_agent = sql_agent
        self._chart_agent = chart_agent

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

        if plan.get("sql") and SkillModule.SQL_QUERYING in config.enabled_skills:
            sql_result = self._sql_agent.run(config.client_id, request.text, context)
            if sql_result.success:
                sql_data = sql_result.data

                if plan.get("chart") and SkillModule.DATA_VISUALIZATION in config.enabled_skills and sql_data:
                    chart_result = self._chart_agent.run(sql_data)
                    if chart_result.success and chart_result.chart_png:
                        charts.append(chart_result.chart_png)

        text = self._generate_response(request, context, sql_data, state.assumptions)

        return Response(
            request_id=str(uuid.uuid4()),
            text=text,
            charts=charts,
            assumptions=state.assumptions,
        )

    def _plan(self, request: Request, context: list[str]) -> dict:
        system = (
            "Determine which capabilities are needed to answer this data request. "
            'Return JSON only: {"sql": true/false, "chart": true/false}'
        )
        user = f"Request: {request.text}\nContext: {chr(10).join(context)}"
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"sql": True, "chart": True}

    def _generate_response(
        self,
        request: Request,
        context: list[str],
        sql_data: dict | None,
        assumptions: list[str],
    ) -> str:
        data_summary = ""
        if sql_data and sql_data.get("rows"):
            rows = sql_data["rows"][:10]
            data_summary = f"\nQuery results (first {len(rows)} rows):\n{json.dumps(rows, default=str)}"

        assumptions_str = ""
        if assumptions:
            assumptions_str = f"\nAssumptions made: {', '.join(assumptions)}\n"

        system = (
            "You are a professional AI data analyst. Write a clear, insightful analysis. "
            "If assumptions were made, state them at the start of your response."
        )
        user = (
            f"Request: {request.text}\n"
            f"{assumptions_str}"
            f"Business context:\n{chr(10).join(context)}"
            f"{data_summary}\n\n"
            "Write the analysis:"
        )
        return self._llm.complete(TaskType.REASONING, system, user)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/orchestrator/test_orchestrator.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/orchestrator.py tests/orchestrator/test_orchestrator.py
git commit -m "feat: orchestrator — full flow from request to response with skill gating"
```

---

## Task 13: Slack Channel

**Files:**
- Create: `src/channels/slack.py`
- Create: `tests/channels/test_slack.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_slack.py
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.channels.slack import SlackChannel
from src.core.models import (
    Request, Channel, ClientConfig, Tier, ClarificationState, Response
)
from src.skill_modules import SkillModuleRegistry


def make_config() -> ClientConfig:
    return ClientConfig(
        client_id="client-1", name="Test Corp", tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor", active_channels=[Channel.SLACK],
    )


def make_slack_event(text: str = "show sales data", thread_ts: str = "123.456") -> dict:
    return {
        "user": "U123",
        "text": f"<@BOTID> {text}",
        "ts": thread_ts,
        "channel": "C001",
        "team": "T001",
    }


def test_normalizes_mention_to_request():
    orchestrator = MagicMock()
    orchestrator.process.return_value = Response(request_id="r1", text="Here is the data.")
    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
    )

    request = channel._build_request(make_slack_event(), "T001")

    assert isinstance(request, Request)
    assert request.channel == Channel.SLACK
    assert request.client_id == "client-1"
    assert "show sales data" in request.text
    assert "<@BOTID>" not in request.text


def test_returns_clarification_questions_as_message():
    orchestrator = MagicMock()
    orchestrator.process.return_value = ClarificationState(
        original_request=MagicMock(),
        rounds=1,
        questions_asked=["Which time period?", "Which region?"],
        is_resolved=False,
    )
    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
    )
    say = MagicMock()
    channel._handle_result(
        result=orchestrator.process.return_value,
        event=make_slack_event(),
        say=say,
    )

    say.assert_called_once()
    call_text = say.call_args[1]["text"]
    assert "Which time period?" in call_text


def test_unknown_workspace_sends_error():
    orchestrator = MagicMock()
    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={},  # no config for this workspace
    )
    say = MagicMock()
    channel._on_unconfigured_workspace(say=say, thread_ts="123.456")
    say.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/channels/test_slack.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
# src/channels/slack.py
from datetime import datetime

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from src.core.models import Channel, ClientConfig, ClarificationState, Request, Response
from src.orchestrator.orchestrator import Orchestrator


class SlackChannel:
    def __init__(
        self,
        bot_token: str,
        signing_secret: str,
        bot_user_id: str,
        orchestrator: Orchestrator,
        client_configs: dict[str, ClientConfig],
    ):
        self._bot_user_id = bot_user_id
        self._orchestrator = orchestrator
        self._client_configs = client_configs
        self._pending: dict[str, ClarificationState] = {}

        self._app = App(token=bot_token, signing_secret=signing_secret)
        self._app.event("app_mention")(self._handle_mention)
        self._handler = SlackRequestHandler(self._app)

    def _handle_mention(self, event: dict, say) -> None:
        team_id = event.get("team")
        config = self._client_configs.get(team_id)
        if not config:
            self._on_unconfigured_workspace(say, event.get("thread_ts") or event.get("ts"))
            return

        thread_ts = event.get("thread_ts") or event.get("ts")
        request = self._build_request(event, team_id)

        pending = self._pending.get(thread_ts)
        if pending and not pending.is_resolved:
            pending.answers_received.append(request.text)

        result = self._orchestrator.process(request, config, pending)
        self._handle_result(result, event, say)

    def _build_request(self, event: dict, team_id: str) -> Request:
        text = event.get("text", "").replace(f"<@{self._bot_user_id}>", "").strip()
        config = self._client_configs.get(team_id)
        return Request(
            channel=Channel.SLACK,
            sender_id=event["user"],
            sender_name=event["user"],
            text=text,
            thread_id=event.get("thread_ts") or event.get("ts"),
            timestamp=datetime.utcnow().isoformat(),
            client_id=config.client_id if config else "unknown",
        )

    def _handle_result(self, result, event: dict, say) -> None:
        thread_ts = event.get("thread_ts") or event.get("ts")

        if isinstance(result, ClarificationState):
            self._pending[thread_ts] = result
            questions = "\n".join(f"• {q}" for q in result.questions_asked[-2:])
            say(text=f"Quick question before I run this:\n{questions}", thread_ts=thread_ts)
            return

        if thread_ts in self._pending:
            del self._pending[thread_ts]

        say(text=result.text, thread_ts=thread_ts)

        for chart_png in result.charts:
            self._app.client.files_upload_v2(
                channel=event["channel"],
                file=chart_png,
                filename="analysis.png",
                thread_ts=thread_ts,
            )

    def _on_unconfigured_workspace(self, say, thread_ts: str) -> None:
        say(
            text="I'm not configured for this workspace yet. Please contact your account manager.",
            thread_ts=thread_ts,
        )

    def get_handler(self) -> SlackRequestHandler:
        return self._handler
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/channels/test_slack.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/channels/slack.py tests/channels/test_slack.py
git commit -m "feat: Slack channel adapter with @mention handler and clarification threading"
```

---

## Task 14: Config & FastAPI Entry Point

**Files:**
- Create: `src/core/config.py`
- Create: `main.py`
- Create: `tests/core/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_config.py
import os
import pytest
from src.core.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("CHROMADB_PERSIST_DIR", ".test-chroma")

    s = Settings()
    assert s.anthropic_api_key == "test-anthropic"
    assert s.openai_api_key == "test-openai"
    assert s.chromadb_persist_dir == ".test-chroma"


def test_settings_has_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "c")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "d")

    s = Settings()
    assert s.chromadb_persist_dir == ".chromadb"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_config.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write Settings**

```python
# src/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str
    chromadb_persist_dir: str = ".chromadb"

    class Config:
        env_file = ".env"
```

- [ ] **Step 4: Write main.py**

```python
# main.py
from fastapi import FastAPI, Request
from src.core.config import Settings
from src.core.llm import LLMRouter
from src.knowledge.vector_store import VectorStore
from src.knowledge.retriever import KnowledgeRetriever
from src.connectors.sql import SQLConnector
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.channels.slack import SlackChannel

settings = Settings()
llm = LLMRouter(
    anthropic_api_key=settings.anthropic_api_key,
    openai_api_key=settings.openai_api_key,
)
store = VectorStore(
    persist_dir=settings.chromadb_persist_dir,
    openai_api_key=settings.openai_api_key,
)
retriever = KnowledgeRetriever(store=store)
clarifier = ClarificationChecker(llm=llm)
chart_agent = ChartAgent()

# Client configs loaded here — in production, load from a config file or DB
client_configs = {}  # workspace_team_id -> ClientConfig

# SQL connector is per-client — wired up at onboarding time
# For now, main.py shows the wiring pattern; real connector injected per deployment

orchestrator = Orchestrator(
    llm=llm,
    retriever=retriever,
    clarifier=clarifier,
    sql_agent=None,   # injected per-client at runtime
    chart_agent=chart_agent,
)

# Slack bot user ID — fetch from Slack API at startup
# app.client.auth_test()["user_id"]
SLACK_BOT_USER_ID = "REPLACE_WITH_BOT_USER_ID"

slack = SlackChannel(
    bot_token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    bot_user_id=SLACK_BOT_USER_ID,
    orchestrator=orchestrator,
    client_configs=client_configs,
)

app = FastAPI()


@app.post("/slack/events")
async def slack_events(req: Request):
    return await slack.get_handler().handle(req)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/core/test_config.py -v
```

Expected: `2 passed`.

- [ ] **Step 6: Run the full test suite**

```bash
pytest -v
```

Expected: All tests pass. Fix any failures before committing.

- [ ] **Step 7: Commit**

```bash
git add src/core/config.py main.py tests/core/test_config.py
git commit -m "feat: settings config and FastAPI entry point"
```

---

## Task 15: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration.py
"""
Full flow: Slack mention → Orchestrator → SQL Agent → Chart Agent → Slack reply.
All external calls (LLM, DB, Slack) are mocked.
"""
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.models import Channel, ClientConfig, Tier, Response, ClarificationState
from src.core.llm import LLMRouter
from src.knowledge.vector_store import VectorStore
from src.knowledge.retriever import KnowledgeRetriever
from src.connectors.sql import SQLConnector
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.skill_modules import SkillModuleRegistry


@pytest.fixture
def full_system(tmp_path, mocker):
    # Mock LLM
    llm = MagicMock(spec=LLMRouter)
    llm.complete.side_effect = [
        '{"is_clear": true, "questions": []}',           # clarifier check
        '{"sql": true, "chart": true}',                   # orchestrator plan
        "Total sales by region for last month:\n- Jakarta: IDR 1,000,000\n- Surabaya: IDR 2,000,000",  # response text
    ]

    # Real vector store (tmp dir), mock embeddings
    mocker.patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction.__call__",
        return_value=[[0.1] * 128],
    )
    store = VectorStore(persist_dir=str(tmp_path), openai_api_key="fake")
    retriever = KnowledgeRetriever(store=store)

    # Mock SQL connector
    connector = MagicMock(spec=SQLConnector)
    connector.get_schema.return_value = {
        "sales": [{"name": "region", "type": "TEXT"}, {"name": "amount", "type": "REAL"}]
    }
    connector.execute.return_value = pd.DataFrame([
        {"region": "Jakarta", "amount": 1_000_000},
        {"region": "Surabaya", "amount": 2_000_000},
    ])

    sql_agent = SQLAgent(llm=llm, connector=connector, store=store)
    chart_agent = ChartAgent()
    clarifier = ClarificationChecker(llm=llm)
    orchestrator = Orchestrator(
        llm=llm,
        retriever=retriever,
        clarifier=clarifier,
        sql_agent=sql_agent,
        chart_agent=chart_agent,
    )

    config = ClientConfig(
        client_id="client-1",
        name="Test Corp",
        tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor",
        active_channels=[Channel.SLACK],
    )

    return orchestrator, config


def make_request(text: str = "show total sales by region for last month"):
    from src.core.models import Request
    return Request(
        channel=Channel.SLACK,
        sender_id="U123",
        sender_name="Budi",
        text=text,
        thread_id="thread-001",
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-1",
    )


def test_clear_request_returns_response_with_chart(full_system):
    orchestrator, config = full_system
    result = orchestrator.process(make_request(), config)

    assert isinstance(result, Response)
    assert "Jakarta" in result.text or "sales" in result.text.lower()
    assert len(result.charts) == 1
    assert result.charts[0][:4] == b"\x89PNG"


def test_ambiguous_request_returns_clarification(full_system, mocker):
    orchestrator, config = full_system
    orchestrator._clarifier._llm.complete.side_effect = [
        '{"is_clear": false, "questions": ["Which time period?", "All regions or specific?"]}',
    ]
    result = orchestrator.process(make_request("show me the data"), config)

    assert isinstance(result, ClarificationState)
    assert result.is_resolved is False
    assert len(result.questions_asked) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_integration.py -v
```

Expected: Tests collected but may fail due to wiring — fix any import or wiring errors until tests pass.

- [ ] **Step 3: Run full test suite**

```bash
pytest -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test — Slack request through Orchestrator to Response"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] On-demand analysis → Task 12 (Orchestrator)
- [x] Clarification loop (max 2 rounds, state assumptions) → Task 11
- [x] Channel Layer (Slack, opt-in) → Task 13
- [x] Self-ticketing → Out of scope for Phase 1 (Jira is Phase 2)
- [x] LLM abstraction layer (multi-provider) → Task 3
- [x] Knowledge Layer (vector store, ingestion, retrieval) → Tasks 5-7
- [x] SQL Connector (read-only) → Task 8
- [x] SQL Agent → Task 9
- [x] Chart Agent (auto-selects chart type) → Task 10
- [x] Skill module registry + tier mapping → Task 4
- [x] Skill gating in Orchestrator → Task 12
- [x] Config & entry point → Task 14
- [x] Integration test → Task 15

**Out of scope for Phase 1 (covered in later phases):**
- Self-ticketing (Jira integration) → Phase 2
- Email / WhatsApp channels → Phase 2
- Scheduled jobs → Phase 3
- ML Agent, Deck Agent → Phase 3
- Anomaly Agent → Phase 4
- Continuous interaction memory logger → Phase 4
