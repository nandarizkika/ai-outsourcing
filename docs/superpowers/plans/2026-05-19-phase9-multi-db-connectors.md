# Phase 9: Multi-Database Connectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PostgreSQL, MySQL, BigQuery, and Snowflake connector support via a `ConnectorFactory` that builds the right SQLAlchemy engine per `connector_type`, wired into `ClientRegistry.get_connector()`.

**Architecture:** `ClientConfig` gains two new optional fields (`connector_type`, `connector_config`). `ConnectorFactory.create(connector_type, connector_config)` builds the correct SQLAlchemy URL and returns the existing `SQLConnector`. `ClientRegistry.get_connector()` calls the factory when `connector_type` is set, falls back to `database_url` otherwise. BigQuery uses Application Default Credentials (ADC) — no credentials in config.

**Tech Stack:** SQLAlchemy, `psycopg2-binary` (Postgres), `pymysql` (MySQL), `sqlalchemy-bigquery` (BigQuery ADC), `snowflake-sqlalchemy` (Snowflake), `pytest`, `unittest.mock.patch`

---

## File Structure

**New files:**
- `src/connectors/factory.py` — `ConnectorFactory` with URL builders per connector type; ~40 lines
- `tests/connectors/__init__.py` — empty, makes `tests/connectors/` a package
- `tests/connectors/test_factory.py` — 8 unit tests (mocked engines, no real DB)

**Modified files:**
- `src/core/models.py` — add `connector_type: Optional[str] = None` and `connector_config: Optional[dict] = None` to `ClientConfig`
- `src/core/client_registry.py` — update `get_connector()` to use `ConnectorFactory` when `connector_type` is set
- `tests/core/test_client_registry.py` — update `_make_config` helper + add 2 new tests
- `pyproject.toml` — add 4 new dependencies

---

### Task 1: Add `connector_type` + `connector_config` to `ClientConfig`

**Files:**
- Modify: `src/core/models.py`
- Modify: `tests/core/test_client_registry.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/core/test_client_registry.py` (after the existing tests):

```python
def test_client_config_accepts_connector_type_and_config(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    config = ClientConfig(
        client_id="c1", name="Client c1", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        connector_type="postgres",
        connector_config={"host": "localhost", "database": "mydb", "user": "u", "password": "p"},
    )
    reg.upsert(config)
    result = reg.get("c1")
    assert result.connector_type == "postgres"
    assert result.connector_config["host"] == "localhost"
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/core/test_client_registry.py::test_client_config_accepts_connector_type_and_config -v
```

Expected: `FAILED` — `ClientConfig` does not accept `connector_type`.

- [ ] **Step 3: Add fields to `ClientConfig` in `src/core/models.py`**

Current `ClientConfig` (lines 57–65):
```python
class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[str]
    account_mode: str
    active_channels: list[Channel]
    database_url: Optional[str] = None
```

Replace with:
```python
class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[str]
    account_mode: str
    active_channels: list[Channel]
    database_url: Optional[str] = None
    connector_type: Optional[str] = None
    connector_config: Optional[dict] = None
```

- [ ] **Step 4: Update `_make_config` helper in `tests/core/test_client_registry.py`**

Current `_make_config` (lines 9–18):
```python
def _make_config(client_id="c1", database_url=None):
    return ClientConfig(
        client_id=client_id,
        name=f"Client {client_id}",
        tier=Tier.BASIC,
        enabled_skills=[],
        account_mode="vendor",
        active_channels=[Channel.SLACK],
        database_url=database_url,
    )
```

Replace with:
```python
def _make_config(client_id="c1", database_url=None, connector_type=None, connector_config=None):
    return ClientConfig(
        client_id=client_id,
        name=f"Client {client_id}",
        tier=Tier.BASIC,
        enabled_skills=[],
        account_mode="vendor",
        active_channels=[Channel.SLACK],
        database_url=database_url,
        connector_type=connector_type,
        connector_config=connector_config,
    )
```

- [ ] **Step 5: Run to verify the new test passes and no regressions**

```bash
.venv/bin/python -m pytest tests/core/test_client_registry.py -v
```

Expected: 10 passed (9 existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add src/core/models.py tests/core/test_client_registry.py
git commit -m "feat: add connector_type and connector_config fields to ClientConfig"
```

---

### Task 2: Create `ConnectorFactory` (TDD)

**Files:**
- Create: `tests/connectors/__init__.py`
- Create: `tests/connectors/test_factory.py`
- Create: `src/connectors/factory.py`

- [ ] **Step 1: Create the test package**

```bash
touch tests/connectors/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/connectors/test_factory.py`:

```python
# tests/connectors/test_factory.py
import pytest
from unittest.mock import patch, MagicMock

from src.connectors.factory import ConnectorFactory
from src.connectors.sql import SQLConnector


def _create(connector_type, connector_config):
    with patch("src.connectors.factory.create_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        conn = ConnectorFactory.create(connector_type, connector_config)
    return conn, mock_engine


def test_postgres_builds_correct_url():
    conn, mock_engine = _create("postgres", {
        "host": "db.example.com", "database": "mydb", "user": "admin", "password": "secret"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "postgresql+psycopg2://admin:secret@db.example.com:5432/mydb"


def test_postgres_default_port():
    _, mock_engine = _create("postgres", {
        "host": "localhost", "database": "mydb", "user": "u", "password": "p"
    })
    assert ":5432/" in mock_engine.call_args[0][0]


def test_mysql_builds_correct_url():
    conn, mock_engine = _create("mysql", {
        "host": "db.example.com", "database": "mydb", "user": "admin", "password": "secret"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "mysql+pymysql://admin:secret@db.example.com:3306/mydb"


def test_mysql_default_port():
    _, mock_engine = _create("mysql", {
        "host": "localhost", "database": "mydb", "user": "u", "password": "p"
    })
    assert ":3306/" in mock_engine.call_args[0][0]


def test_bigquery_builds_correct_url():
    conn, mock_engine = _create("bigquery", {
        "project_id": "my-project", "dataset": "analytics"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "bigquery://my-project/analytics"


def test_snowflake_builds_correct_url():
    conn, mock_engine = _create("snowflake", {
        "account": "myaccount", "user": "admin", "password": "secret",
        "warehouse": "COMPUTE_WH", "database": "mydb"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "snowflake://admin:secret@myaccount/mydb/PUBLIC?warehouse=COMPUTE_WH"


def test_snowflake_custom_schema():
    _, mock_engine = _create("snowflake", {
        "account": "myaccount", "user": "admin", "password": "secret",
        "warehouse": "COMPUTE_WH", "database": "mydb", "schema": "RAW"
    })
    assert "/RAW?" in mock_engine.call_args[0][0]


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown connector_type"):
        ConnectorFactory.create("oracle", {})
```

- [ ] **Step 3: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/connectors/test_factory.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.connectors.factory'`

- [ ] **Step 4: Create `src/connectors/factory.py`**

```python
# src/connectors/factory.py
from sqlalchemy import create_engine

from src.connectors.sql import SQLConnector


def _postgres_url(cfg: dict) -> str:
    return (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )


def _mysql_url(cfg: dict) -> str:
    return (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 3306)}/{cfg['database']}"
    )


def _bigquery_url(cfg: dict) -> str:
    return f"bigquery://{cfg['project_id']}/{cfg['dataset']}"


def _snowflake_url(cfg: dict) -> str:
    schema = cfg.get("schema", "PUBLIC")
    return (
        f"snowflake://{cfg['user']}:{cfg['password']}"
        f"@{cfg['account']}/{cfg['database']}/{schema}"
        f"?warehouse={cfg['warehouse']}"
    )


_BUILDERS = {
    "postgres": _postgres_url,
    "mysql": _mysql_url,
    "bigquery": _bigquery_url,
    "snowflake": _snowflake_url,
}


class ConnectorFactory:
    @staticmethod
    def create(connector_type: str, connector_config: dict) -> SQLConnector:
        builder = _BUILDERS.get(connector_type)
        if builder is None:
            raise ValueError(f"Unknown connector_type: {connector_type!r}")
        url = builder(connector_config)
        engine = create_engine(url)
        return SQLConnector(connection_url=url, engine=engine)
```

- [ ] **Step 5: Run to verify all 8 tests pass**

```bash
.venv/bin/python -m pytest tests/connectors/test_factory.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: ~243 passed, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add src/connectors/factory.py tests/connectors/__init__.py tests/connectors/test_factory.py
git commit -m "feat: add ConnectorFactory with Postgres, MySQL, BigQuery, Snowflake support"
```

---

### Task 3: Wire `ConnectorFactory` into `ClientRegistry.get_connector()`

**Files:**
- Modify: `src/core/client_registry.py`
- Modify: `tests/core/test_client_registry.py`

- [ ] **Step 1: Write the failing tests**

Append these two tests to `tests/core/test_client_registry.py`:

```python
def test_get_connector_uses_connector_type_when_set(tmp_path):
    from unittest.mock import patch, MagicMock
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    config = _make_config(
        "c1",
        connector_type="postgres",
        connector_config={"host": "localhost", "database": "mydb", "user": "u", "password": "p"},
    )
    reg.upsert(config)
    mock_connector = MagicMock()
    with patch("src.core.client_registry.ConnectorFactory.create", return_value=mock_connector) as mock_create:
        result = reg.get_connector("c1")
    assert result is mock_connector
    mock_create.assert_called_once_with(
        "postgres", {"host": "localhost", "database": "mydb", "user": "u", "password": "p"}
    )


def test_get_connector_connector_type_takes_priority_over_database_url(tmp_path):
    from unittest.mock import patch, MagicMock
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    config = _make_config(
        "c1",
        database_url="sqlite:///:memory:",
        connector_type="mysql",
        connector_config={"host": "localhost", "database": "mydb", "user": "u", "password": "p"},
    )
    reg.upsert(config)
    mock_connector = MagicMock()
    with patch("src.core.client_registry.ConnectorFactory.create", return_value=mock_connector):
        result = reg.get_connector("c1")
    assert result is mock_connector
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/core/test_client_registry.py::test_get_connector_uses_connector_type_when_set tests/core/test_client_registry.py::test_get_connector_connector_type_takes_priority_over_database_url -v
```

Expected: both FAIL — `get_connector` ignores `connector_type`.

- [ ] **Step 3: Update `get_connector()` in `src/core/client_registry.py`**

Current imports (top of file):
```python
import json
import threading

from src.connectors.sql import SQLConnector
from src.core.models import ClientConfig
```

Replace with:
```python
import json
import threading

from src.connectors.factory import ConnectorFactory
from src.connectors.sql import SQLConnector
from src.core.models import ClientConfig
```

Current `get_connector()` method (lines 51–61):
```python
    def get_connector(self, client_id: str) -> SQLConnector | None:
        with self._lock:
            raw = self._load().get(client_id)
            if raw is None:
                return None
            db_url = raw.get("database_url")
            if not db_url:
                return None
            if client_id not in self._connectors:
                self._connectors[client_id] = SQLConnector(connection_url=db_url)
            return self._connectors[client_id]
```

Replace with:
```python
    def get_connector(self, client_id: str) -> SQLConnector | None:
        with self._lock:
            raw = self._load().get(client_id)
            if raw is None:
                return None
            if client_id not in self._connectors:
                connector_type = raw.get("connector_type")
                connector_config = raw.get("connector_config")
                db_url = raw.get("database_url")
                if connector_type and connector_config is not None:
                    self._connectors[client_id] = ConnectorFactory.create(connector_type, connector_config)
                elif db_url:
                    self._connectors[client_id] = SQLConnector(connection_url=db_url)
                else:
                    return None
            return self._connectors[client_id]
```

- [ ] **Step 4: Run to verify new tests pass**

```bash
.venv/bin/python -m pytest tests/core/test_client_registry.py -v
```

Expected: 12 passed (10 existing + 2 new).

- [ ] **Step 5: Run full suite for regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: ~245 passed, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/core/client_registry.py tests/core/test_client_registry.py
git commit -m "feat: wire ConnectorFactory into ClientRegistry.get_connector()"
```

---

### Task 4: Add dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Check current dependencies section**

```bash
grep -n "dependencies" pyproject.toml | head -5
```

- [ ] **Step 2: Add the 4 new packages**

In `pyproject.toml`, find the `dependencies` list and add the four new entries alongside the existing ones:

```toml
"psycopg2-binary>=2.9",
"pymysql>=1.1",
"sqlalchemy-bigquery>=1.9",
"snowflake-sqlalchemy>=1.5",
```

- [ ] **Step 3: Install new dependencies**

```bash
.venv/bin/pip install psycopg2-binary pymysql sqlalchemy-bigquery snowflake-sqlalchemy 2>&1 | tail -5
```

- [ ] **Step 4: Run full suite to confirm nothing broke**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: ~245 passed, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add psycopg2-binary, pymysql, sqlalchemy-bigquery, snowflake-sqlalchemy dependencies"
```

---

## Self-Review

**Spec coverage:**
- ✅ PostgreSQL connector — Task 2 (`_postgres_url`)
- ✅ MySQL connector — Task 2 (`_mysql_url`)
- ✅ BigQuery connector with ADC — Task 2 (`_bigquery_url`, no credentials in config)
- ✅ Snowflake connector — Task 2 (`_snowflake_url`)
- ✅ `connector_type` + `connector_config` on `ClientConfig` — Task 1
- ✅ `ConnectorFactory` — Task 2
- ✅ `ClientRegistry.get_connector()` uses factory — Task 3
- ✅ `connector_type` takes priority over `database_url` — Task 3 (`test_get_connector_connector_type_takes_priority_over_database_url`)
- ✅ Backwards compatibility (`database_url` fallback) — Task 3 (existing tests still pass)
- ✅ Dependencies — Task 4

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `ConnectorFactory.create(connector_type: str, connector_config: dict) -> SQLConnector` — matches usage in `get_connector()` and test mocks.
- `_make_config(connector_type=None, connector_config=None)` — matches `ClientConfig` field names exactly.
- `patch("src.core.client_registry.ConnectorFactory.create", ...)` — patches the name as imported in `client_registry.py`. Consistent with the import added in Task 3 Step 3.
