from unittest.mock import MagicMock
import pytest

from src.agents.ml_agent import MLAgent
from src.core.models import AgentResult


def _make_agent(llm_response="Forecast: revenue will grow 12% over the next 4 weeks."):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = llm_response
    return MLAgent(llm=mock_llm)


def _time_series_data():
    return {
        "columns": ["week", "revenue"],
        "rows": [
            {"week": 1, "revenue": 100},
            {"week": 2, "revenue": 110},
            {"week": 3, "revenue": 108},
            {"week": 4, "revenue": 115},
            {"week": 5, "revenue": 120},
            {"week": 6, "revenue": 118},
            {"week": 7, "revenue": 125},
            {"week": 8, "revenue": 130},
            {"week": 9, "revenue": 128},
            {"week": 10, "revenue": 135},
        ],
    }


def test_forecast_returns_chart_and_text():
    agent = _make_agent()
    result = agent.run(
        client_id="client1",
        request="Forecast next 4 weeks of revenue",
        data=_time_series_data(),
    )
    assert result.success is True
    assert result.chart_png is not None
    assert len(result.chart_png) > 0
    assert result.data is not None
    assert "forecast" in result.data


def test_forecast_interpretation_comes_from_llm():
    agent = _make_agent(llm_response="Revenue will increase by 15%.")
    result = agent.run(
        client_id="client1",
        request="Forecast revenue",
        data=_time_series_data(),
    )
    assert result.success is True
    assert result.data["interpretation"] == "Revenue will increase by 15%."


def test_insufficient_data_returns_error():
    agent = _make_agent()
    result = agent.run(
        client_id="client1",
        request="Forecast revenue",
        data={"columns": ["revenue"], "rows": [{"revenue": 100}]},
    )
    assert result.success is False
    assert result.error is not None
    assert "insufficient" in result.error.lower() or "not enough" in result.error.lower()


def test_no_numeric_column_returns_error():
    agent = _make_agent()
    result = agent.run(
        client_id="client1",
        request="Forecast",
        data={
            "columns": ["name", "category"],
            "rows": [{"name": "A", "category": "X"}, {"name": "B", "category": "Y"}],
        },
    )
    assert result.success is False
    assert result.error is not None
