import json
import pytest
from unittest.mock import MagicMock

from src.agents.anomaly_agent import AnomalyAgent
from src.core.models import AgentResult


def _agent_with_rules(*rule_jsons):
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [json.dumps(r) for r in rule_jsons]
    return AnomalyAgent(retriever=mock_retriever)


def _agent_no_rules():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    return AnomalyAgent(retriever=mock_retriever)


def test_hard_rule_detects_breach():
    agent = _agent_with_rules(
        {"metric": "churn_rate", "operator": ">", "threshold": 0.1, "severity": "critical"}
    )
    result = agent.run(
        client_id="c1",
        data={"columns": ["churn_rate"], "rows": [{"churn_rate": 0.15}]},
        mode="hard",
    )
    assert result.success is True
    anomalies = result.data["anomalies"]
    assert len(anomalies) == 1
    assert anomalies[0]["metric"] == "churn_rate"
    assert anomalies[0]["severity"] == "critical"
    assert anomalies[0]["mode"] == "hard_rule"


def test_hard_rule_no_breach():
    agent = _agent_with_rules(
        {"metric": "churn_rate", "operator": ">", "threshold": 0.2, "severity": "warning"}
    )
    result = agent.run(
        client_id="c1",
        data={"columns": ["churn_rate"], "rows": [{"churn_rate": 0.15}]},
        mode="hard",
    )
    assert result.success is True
    assert result.data["anomalies"] == []


def test_hard_rule_all_operators():
    for op, val, threshold, should_flag in [
        (">", 5, 4, True), (">", 4, 5, False),
        (">=", 5, 5, True), (">=", 4, 5, False),
        ("<", 3, 4, True), ("<", 5, 4, False),
        ("<=", 4, 4, True), ("<=", 5, 4, False),
    ]:
        agent = _agent_with_rules({"metric": "m", "operator": op, "threshold": threshold})
        result = agent.run("c1", {"columns": ["m"], "rows": [{"m": val}]}, mode="hard")
        assert (len(result.data["anomalies"]) > 0) == should_flag, f"op={op} val={val} threshold={threshold}"


def test_invalid_rule_json_skipped():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        "not valid json",
        json.dumps({"metric": "revenue", "operator": ">", "threshold": 1000}),
    ]
    agent = AnomalyAgent(retriever=mock_retriever)
    result = agent.run(
        client_id="c1",
        data={"columns": ["revenue"], "rows": [{"revenue": 1500}]},
        mode="hard",
    )
    assert result.success is True
    assert len(result.data["anomalies"]) == 1


def test_statistical_detects_outlier():
    agent = _agent_no_rules()
    data = {
        "columns": ["revenue"],
        "rows": [
            {"revenue": 100}, {"revenue": 102}, {"revenue": 98},
            {"revenue": 99}, {"revenue": 101}, {"revenue": 500},
        ],
    }
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.success is True
    anomalies = result.data["anomalies"]
    assert any(a["value"] == 500 and a["metric"] == "revenue" for a in anomalies)


def test_statistical_no_anomaly_in_uniform_data():
    agent = _agent_no_rules()
    data = {
        "columns": ["revenue"],
        "rows": [{"revenue": v} for v in [100, 102, 98, 99, 101, 103]],
    }
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.data["anomalies"] == []


def test_statistical_skips_non_numeric_columns():
    agent = _agent_no_rules()
    data = {
        "columns": ["name", "revenue"],
        "rows": [
            {"name": "Alice", "revenue": 100},
            {"name": "Bob", "revenue": 102},
            {"name": "Charlie", "revenue": 99},
        ],
    }
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.success is True
    assert all(a["metric"] != "name" for a in result.data["anomalies"])


def test_both_mode_runs_both():
    agent = _agent_with_rules(
        {"metric": "churn_rate", "operator": ">", "threshold": 0.05, "severity": "warning"}
    )
    data = {
        "columns": ["churn_rate", "revenue"],
        "rows": [
            {"churn_rate": 0.10, "revenue": 100},
            {"churn_rate": 0.09, "revenue": 102},
            {"churn_rate": 0.08, "revenue": 98},
            {"churn_rate": 0.07, "revenue": 99},
            {"churn_rate": 0.11, "revenue": 101},
            {"churn_rate": 0.06, "revenue": 103},
            {"churn_rate": 0.09, "revenue": 500},
        ],
    }
    result = agent.run(client_id="c1", data=data, mode="both")
    modes = {a["mode"] for a in result.data["anomalies"]}
    assert "hard_rule" in modes
    assert "statistical" in modes


def test_insufficient_data_for_statistical_returns_no_anomalies():
    agent = _agent_no_rules()
    data = {"columns": ["revenue"], "rows": [{"revenue": 100}, {"revenue": 200}]}
    result = agent.run(client_id="c1", data=data, mode="statistical")
    assert result.success is True
    assert result.data["anomalies"] == []
