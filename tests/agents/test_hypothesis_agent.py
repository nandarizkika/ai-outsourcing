# tests/agents/test_hypothesis_agent.py
import numpy as np
from unittest.mock import MagicMock

from src.agents.hypothesis_agent import HypothesisAgent


def _make_agent():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Group B shows significantly higher values."
    return HypothesisAgent(llm=mock_llm)


def _two_group_continuous():
    rng = np.random.default_rng(42)
    rows_a = [{"group": "A", "metric": float(v)} for v in rng.normal(10, 1, 30)]
    rows_b = [{"group": "B", "metric": float(v)} for v in rng.normal(15, 1, 30)]
    return {"columns": ["group", "metric"], "rows": rows_a + rows_b}


def _two_group_binary():
    rows = (
        [{"group": "A", "converted": 1} for _ in range(20)] +
        [{"group": "A", "converted": 0} for _ in range(10)] +
        [{"group": "B", "converted": 1} for _ in range(25)] +
        [{"group": "B", "converted": 0} for _ in range(5)]
    )
    return {"columns": ["group", "converted"], "rows": rows}


def test_ttest_detects_significant_difference():
    agent = _make_agent()
    result = agent.run("c1", "Is there a significant difference?", _two_group_continuous())
    assert result.success is True
    assert result.data["test"] == "t_test"
    assert result.data["p_value"] < 0.05
    assert result.data["significant"] is True


def test_returns_statistic_and_groups():
    agent = _make_agent()
    result = agent.run("c1", "Compare groups", _two_group_continuous())
    assert "statistic" in result.data
    assert "groups" in result.data
    assert len(result.data["groups"]) == 2


def test_binary_metric_uses_chi2():
    agent = _make_agent()
    result = agent.run("c1", "Compare conversion rates", _two_group_binary())
    assert result.success is True
    assert result.data["test"] == "chi2"


def test_llm_interpretation_included():
    agent = _make_agent()
    result = agent.run("c1", "Compare groups", _two_group_continuous())
    assert result.data["interpretation"] == "Group B shows significantly higher values."


def test_too_few_rows_returns_error():
    agent = _make_agent()
    data = {"columns": ["group", "metric"], "rows": [{"group": "A", "metric": 1.0}]}
    result = agent.run("c1", "Compare", data)
    assert result.success is False
    assert result.error is not None


def test_no_group_column_returns_error():
    agent = _make_agent()
    data = {
        "columns": ["x", "y"],
        "rows": [{"x": float(i), "y": float(i * 2)} for i in range(10)],
    }
    result = agent.run("c1", "Compare", data)
    assert result.success is False
