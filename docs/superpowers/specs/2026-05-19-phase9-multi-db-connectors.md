# Phase 9: Multi-Database Connectors

**Date:** 2026-05-19

---

## Goal

Add PostgreSQL, MySQL, BigQuery, and Snowflake connector support to the `ClientRegistry` and `SQLAgent` pipeline, via a `ConnectorFactory` that builds the right SQLAlchemy engine per connector type.

---

## Architecture

The existing `SQLConnector` already uses SQLAlchemy, which natively supports all four databases. No new connector classes are needed — a `ConnectorFactory` builds the right SQLAlchemy engine from `connector_type` + `connector_config` and returns the existing `SQLConnector`.

Two new fields on `ClientConfig` define the connector:
- `connector_type: Optional[str]` — `"postgres"`, `"mysql"`, `"bigquery"`, `"snowflake"`
- `connector_config: Optional[dict]` — type-specific connection parameters

The existing `database_url` field remains as a fallback for raw SQLAlchemy URL connections (backwards compatible).

BigQuery uses **Application Default Credentials (ADC)** — no credentials JSON stored in config. One-time setup: `gcloud auth application-default login`.

---

## Component Designs

### 1. `ClientConfig` changes (`src/core/models.py`)

Add two optional fields:

```python
connector_type: Optional[str] = None
connector_config: Optional[dict] = None
```

Existing `database_url` remains unchanged.

---

### 2. `src/connectors/factory.py`

```python
from src.connectors.sql import SQLConnector
from sqlalchemy import create_engine


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
    return (
        f"snowflake://{cfg['user']}:{cfg['password']}"
        f"@{cfg['account']}/{cfg['database']}/{cfg.get('schema', 'PUBLIC')}"
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

---

### 3. `ClientRegistry.get_connector()` (`src/core/client_registry.py`)

Current implementation creates a `SQLConnector` only when `config.database_url` is set.

Updated logic (priority: `connector_type` > `database_url`):

```python
def get_connector(self, client_id: str):
    with self._lock:
        if client_id in self._connectors:
            return self._connectors[client_id]
        config = self.get(client_id)
        if config is None:
            return None
        connector = None
        if config.connector_type and config.connector_config is not None:
            connector = ConnectorFactory.create(config.connector_type, config.connector_config)
        elif config.database_url:
            connector = SQLConnector(connection_url=config.database_url)
        if connector is not None:
            self._connectors[client_id] = connector
        return connector
```

Cache eviction on `upsert()` already exists — no change needed there.

---

### 4. `pyproject.toml`

Add to dependencies:

```toml
"psycopg2-binary>=2.9",
"pymysql>=1.1",
"sqlalchemy-bigquery>=1.9",
"snowflake-sqlalchemy>=1.5",
```

---

## Per-Connector Config Reference

| `connector_type` | Required fields in `connector_config` | Optional fields |
|------------------|---------------------------------------|-----------------|
| `"postgres"` | `host`, `database`, `user`, `password` | `port` (default 5432) |
| `"mysql"` | `host`, `database`, `user`, `password` | `port` (default 3306) |
| `"bigquery"` | `project_id`, `dataset` | — (ADC handles auth) |
| `"snowflake"` | `account`, `user`, `password`, `warehouse`, `database` | `schema` (default `PUBLIC`) |

**BigQuery setup:** Run once before starting the service:
```bash
gcloud auth application-default login
```

---

## File Structure

**New files:**
- `src/connectors/factory.py` — `ConnectorFactory` with URL builders per connector type
- `tests/connectors/test_factory.py` — unit tests (mocked SQLAlchemy engines, no real DB)

**Modified files:**
- `src/core/models.py` — add `connector_type`, `connector_config` to `ClientConfig`
- `src/core/client_registry.py` — `get_connector()` uses `ConnectorFactory` when `connector_type` is set
- `pyproject.toml` — add 4 new dependencies

---

## Testing Strategy

All tests use mocked SQLAlchemy engines — no real database connections required.

**`tests/connectors/test_factory.py`** (unit tests, ~8 tests):
- `test_postgres_builds_correct_url`
- `test_postgres_default_port`
- `test_mysql_builds_correct_url`
- `test_mysql_default_port`
- `test_bigquery_builds_correct_url`
- `test_snowflake_builds_correct_url`
- `test_snowflake_default_schema`
- `test_unknown_type_raises_value_error`

**`tests/core/test_client_registry.py`** (extend existing, ~3 new tests):
- `test_get_connector_uses_connector_type_when_set`
- `test_get_connector_falls_back_to_database_url`
- `test_get_connector_returns_none_when_neither_set`

---

## Self-Review

**Placeholder scan:** No TBDs or incomplete sections.

**Internal consistency:**
- `ConnectorFactory.create()` always returns a `SQLConnector` — the rest of the pipeline (`SQLAgent`, `Orchestrator`) needs no changes.
- Cache eviction in `upsert()` already clears `self._connectors[client_id]` — covers `connector_type`/`connector_config` changes too.
- `connector_type` takes priority over `database_url` — documented in both the component design and the config reference.

**Scope check:** 2 new files, 3 modified — appropriate for a single phase.

**Ambiguity check:**
- BigQuery ADC: explicit. The service must be started in an environment where `gcloud auth application-default login` has been run.
- Snowflake `schema` default is `PUBLIC` — standard Snowflake default.
- Port defaults (5432 for Postgres, 3306 for MySQL) are industry standard.
