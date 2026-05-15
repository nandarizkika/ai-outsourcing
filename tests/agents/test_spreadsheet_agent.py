# tests/agents/test_spreadsheet_agent.py
import io
import json
from unittest.mock import MagicMock

import pandas as pd

from src.agents.spreadsheet_agent import SpreadsheetAgent
from src.core.models import AgentResult, Channel, ClientConfig, SkillModule, Tier


def _make_config():
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ADVANCED,
        enabled_skills=[SkillModule.SPREADSHEET_ANALYSIS],
        account_mode="vendor", active_channels=[Channel.SLACK],
    )


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _agent(ops_json: str = '{"operation": "describe"}'):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = ops_json
    return SpreadsheetAgent(llm=mock_llm)


def test_reads_csv_and_returns_agent_result():
    df = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100, 200, 300]})
    agent = _agent('{"operation": "describe"}')
    result = agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="Summarize revenue",
        config=_make_config(),
    )
    assert result.success is True
    assert result.data is not None
    assert "rows" in result.data
    assert "columns" in result.data


def test_reads_xlsx_and_returns_agent_result():
    df = pd.DataFrame({"region": ["X", "Y"], "sales": [500, 600]})
    agent = _agent('{"operation": "describe"}')
    result = agent.run(
        file_bytes=_xlsx_bytes(df),
        filename="data.xlsx",
        question="Summarize sales",
        config=_make_config(),
    )
    assert result.success is True
    assert "rows" in result.data


def test_groupby_operation():
    df = pd.DataFrame({
        "region": ["A", "A", "B", "B"],
        "revenue": [100, 150, 200, 250],
    })
    ops = json.dumps({"operation": "groupby", "by": "region", "agg": {"revenue": "sum"}})
    agent = _agent(ops)
    result = agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="Total revenue by region",
        config=_make_config(),
    )
    assert result.success is True
    rows = result.data["rows"]
    regions = {r["region"] for r in rows}
    assert regions == {"A", "B"}


def test_filter_operation():
    df = pd.DataFrame({"region": ["A", "B", "C"], "revenue": [100, 200, 300]})
    ops = json.dumps({"operation": "filter", "column": "revenue", "operator": ">", "value": 150})
    agent = _agent(ops)
    result = agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="High revenue regions",
        config=_make_config(),
    )
    assert result.success is True
    assert all(r["revenue"] > 150 for r in result.data["rows"])


def test_unsupported_format_returns_failure():
    agent = _agent()
    result = agent.run(
        file_bytes=b"not a real file",
        filename="data.parquet",
        question="Summarize",
        config=_make_config(),
    )
    assert result.success is False
    assert result.error is not None


def test_llm_called_with_question_and_columns():
    df = pd.DataFrame({"month": [1, 2, 3], "revenue": [100, 200, 300]})
    agent = _agent('{"operation": "describe"}')
    agent.run(
        file_bytes=_csv_bytes(df),
        filename="data.csv",
        question="Show revenue trend",
        config=_make_config(),
    )
    call_args = agent._llm.complete.call_args
    prompt = call_args[0][2]  # user prompt (3rd positional arg)
    assert "Show revenue trend" in prompt
    assert "month" in prompt or "revenue" in prompt
