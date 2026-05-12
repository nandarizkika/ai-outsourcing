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


def test_blocks_delete(connector):
    with pytest.raises(ValueError, match="Read-only"):
        connector.execute("DELETE FROM sales WHERE id = 1")


def test_blocks_create(connector):
    with pytest.raises(ValueError, match="Read-only"):
        connector.execute("CREATE TABLE foo (id INTEGER)")


def test_execute_empty_result(connector):
    df = connector.execute("SELECT * FROM sales WHERE id = 9999")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_execute_empty_query(connector):
    with pytest.raises(ValueError, match="empty"):
        connector.execute("   ")


def test_get_schema_has_type_field(connector):
    schema = connector.get_schema()
    col = schema["sales"][0]
    assert "type" in col
    assert isinstance(col["type"], str)
