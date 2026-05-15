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


def test_funnel_mode_adds_hint_to_system_prompt():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT stage, COUNT(*) FROM funnel GROUP BY stage"
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {}
    mock_connector.execute.return_value = pd.DataFrame(
        {"stage": ["A", "B"], "count": [100, 80]}
    )
    mock_store = MagicMock()
    agent = SQLAgent(llm=mock_llm, connector=mock_connector, store=mock_store)
    agent.run("c1", "show funnel", [], analysis_mode="funnel")
    system_prompt = mock_llm.complete.call_args[0][1]
    assert "funnel" in system_prompt.lower()
    assert "conversion" in system_prompt.lower()


def test_cohort_mode_adds_hint_to_system_prompt():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT cohort, period, retention FROM cohorts"
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {}
    mock_connector.execute.return_value = pd.DataFrame(
        {"cohort": ["Jan"], "period": [1], "retention": [0.8]}
    )
    mock_store = MagicMock()
    agent = SQLAgent(llm=mock_llm, connector=mock_connector, store=mock_store)
    agent.run("c1", "show cohort retention", [], analysis_mode="cohort")
    system_prompt = mock_llm.complete.call_args[0][1]
    assert "cohort" in system_prompt.lower()
    assert "retention" in system_prompt.lower()


def test_no_analysis_mode_unchanged():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SELECT 1"
    mock_connector = MagicMock()
    mock_connector.get_schema.return_value = {}
    mock_connector.execute.return_value = pd.DataFrame({"x": [1]})
    mock_store = MagicMock()
    agent = SQLAgent(llm=mock_llm, connector=mock_connector, store=mock_store)
    agent.run("c1", "show revenue", [])
    system_prompt = mock_llm.complete.call_args[0][1]
    assert "funnel" not in system_prompt.lower()
    assert "cohort" not in system_prompt.lower()
