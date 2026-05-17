# tests/agents/test_report_agent.py
from unittest.mock import MagicMock

from src.agents.report_agent import ReportAgent


def _make_agent(md_response="## Executive Summary\n- Revenue up\n\n## Recommendation\nActnow."):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = md_response
    return ReportAgent(llm=mock_llm)


def _base_data(**kwargs):
    return {
        "analysis_text": "Revenue grew 10% in Q2.",
        "sql_rows": [{"month": "Jan", "revenue": 1000}, {"month": "Feb", "revenue": 1100}],
        "anomalies": [],
        **kwargs,
    }


def test_report_returns_markdown_and_html():
    agent = _make_agent()
    result = agent.run("c1", "Generate a report", _base_data())
    assert result.success is True
    assert "markdown" in result.data
    assert "html" in result.data
    assert len(result.data["markdown"]) > 0
    assert len(result.data["html"]) > 0


def test_html_output_contains_html_structure():
    agent = _make_agent()
    result = agent.run("c1", "Generate a report", _base_data())
    assert "<" in result.data["html"]


def test_empty_sql_rows_still_produces_report():
    agent = _make_agent()
    result = agent.run("c1", "Generate a report", _base_data(sql_rows=[]))
    assert result.success is True
    assert result.data["markdown"] is not None


def test_anomalies_included_in_llm_prompt():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "## Executive Summary\n- ok"
    agent = ReportAgent(llm=mock_llm)
    data = _base_data(anomalies=[{"metric": "revenue", "description": "Drop detected", "value": 50.0, "severity": "warning", "mode": "hard_rule"}])
    agent.run("c1", "report", data)
    call_args = mock_llm.complete.call_args
    user_prompt = call_args[0][2]
    assert "Drop detected" in user_prompt


def test_llm_failure_returns_error():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM down")
    agent = ReportAgent(llm=mock_llm)
    result = agent.run("c1", "report", _base_data())
    assert result.success is False
    assert result.error is not None
