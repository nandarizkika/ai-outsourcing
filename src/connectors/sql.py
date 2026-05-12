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
