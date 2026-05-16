# tests/agents/test_ab_agent.py
import numpy as np
from unittest.mock import MagicMock

from src.agents.ab_agent import ABTestingAgent


def _make_agent():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Ship the variant."
    return ABTestingAgent(llm=mock_llm)


def _binary_ab_data():
    rows = (
        [{"variant": "control", "converted": 1} for _ in range(100)] +
        [{"variant": "control", "converted": 0} for _ in range(900)] +
        [{"variant": "treatment", "converted": 1} for _ in range(150)] +
        [{"variant": "treatment", "converted": 0} for _ in range(850)]
    )
    return {"columns": ["variant", "converted"], "rows": rows}


def _continuous_ab_data():
    rng = np.random.default_rng(42)
    rows = (
        [{"variant": "control", "revenue": float(v)} for v in rng.normal(50, 5, 100)] +
        [{"variant": "treatment", "revenue": float(v)} for v in rng.normal(58, 5, 100)]
    )
    return {"columns": ["variant", "revenue"], "rows": rows}


def test_binary_ab_returns_proportions_ztest():
    agent = _make_agent()
    result = agent.run("c1", "A/B test results", _binary_ab_data())
    assert result.success is True
    assert result.data["test"] == "proportions_ztest"


def test_continuous_ab_returns_ttest():
    agent = _make_agent()
    result = agent.run("c1", "Compare revenue", _continuous_ab_data())
    assert result.success is True
    assert result.data["test"] == "t_test"


def test_returns_positive_uplift_for_better_variant():
    agent = _make_agent()
    result = agent.run("c1", "A/B results", _binary_ab_data())
    assert "uplift" in result.data
    assert result.data["uplift"] > 0  # treatment (15%) > control (10%)


def test_continuous_significant_result():
    agent = _make_agent()
    result = agent.run("c1", "Revenue test", _continuous_ab_data())
    assert result.data["significant"] is True


def test_returns_control_and_variant_means():
    agent = _make_agent()
    result = agent.run("c1", "A/B test", _binary_ab_data())
    assert "control_mean" in result.data
    assert "variant_mean" in result.data
    assert abs(result.data["control_mean"] - 0.10) < 0.01
    assert abs(result.data["variant_mean"] - 0.15) < 0.01


def test_llm_recommendation_included():
    agent = _make_agent()
    result = agent.run("c1", "A/B test", _binary_ab_data())
    assert result.data["recommendation"] == "Ship the variant."


def test_no_control_group_returns_error():
    agent = _make_agent()
    data = {
        "columns": ["variant", "metric"],
        "rows": [{"variant": "A", "metric": float(i)} for i in range(10)] +
                [{"variant": "B", "metric": float(i)} for i in range(10)],
    }
    result = agent.run("c1", "test", data)
    assert result.success is False
    assert result.error is not None
