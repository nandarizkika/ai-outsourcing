# tests/agents/test_sql_agent_store_optional.py
from unittest.mock import MagicMock
from sqlalchemy import create_engine, text

from src.agents.sql_agent import SQLAgent
from src.connectors.sql import SQLConnector


def _make_connector():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO users VALUES (1, 'Alice')"))
        conn.commit()
    return SQLConnector(connection_url="sqlite:///:memory:", engine=engine)


def test_sql_agent_works_without_store():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT * FROM users"
    connector = _make_connector()
    agent = SQLAgent(llm=mock_llm, connector=connector)
    result = agent.run("c1", "Show me users", [])
    assert result.success is True
    assert result.data is not None


def test_sql_agent_does_not_call_store_when_none():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT * FROM users"
    connector = _make_connector()
    mock_store = MagicMock()
    agent_no_store = SQLAgent(llm=mock_llm, connector=connector, store=None)
    agent_no_store.run("c1", "Show me users", [])
    mock_store.add.assert_not_called()


def test_sql_agent_calls_store_when_provided():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT * FROM users"
    connector = _make_connector()
    mock_store = MagicMock()
    agent = SQLAgent(llm=mock_llm, connector=connector, store=mock_store)
    agent.run("c1", "Show me users", [])
    mock_store.add.assert_called_once()
