# src/connectors/sql.py
from sqlalchemy import create_engine, text, inspect, Engine
import pandas as pd


_READ_ONLY_PREFIXES = frozenset({"SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE"})


class SQLConnector:
    def __init__(self, connection_url: str, engine: Engine | None = None):
        self._engine = engine or create_engine(connection_url)

    def get_schema(self) -> dict[str, list[dict[str, str]]]:
        inspector = inspect(self._engine)
        return {
            table: [
                {"name": col["name"], "type": str(col["type"])}
                for col in inspector.get_columns(table)
            ]
            for table in inspector.get_table_names()
        }

    def execute(self, query: str) -> pd.DataFrame:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        first_word = query.strip().split()[0].upper()
        if first_word not in _READ_ONLY_PREFIXES:
            raise ValueError("Read-only access only. Only SELECT/WITH/EXPLAIN/SHOW/DESCRIBE queries are permitted.")
        with self._engine.connect() as conn:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
