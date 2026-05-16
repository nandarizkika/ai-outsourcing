# tests/agents/test_analyst_agent.py
import json
import pytest
from unittest.mock import MagicMock

from src.agents.analyst_agent import AnalystAgent
from src.core.models import (
    AgentResult, AnalystResult, Channel, ClientConfig, Request, SkillModule, StepRecord, Tier,
)


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text="Why did churn spike in Q2?"):
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Ana",
        text=text, timestamp="2026-05-14T00:00:00", client_id="c1",
    )


def _make_agent(llm_responses: list[str]):
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = llm_responses
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"rows": [{"churn": 0.15}], "columns": ["churn"], "query": "SELECT churn FROM m"},
    )
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    return AnalystAgent(llm=mock_llm, sql_agent=mock_sql, retriever=mock_retriever)


def _step_response(tool="DONE", tool_input=None, thought="Analysis complete"):
    return json.dumps({
        "thought": thought,
        "tool": tool,
        "tool_input": tool_input or {},
    })


def test_loop_terminates_when_done_returned():
    agent = _make_agent([
        _step_response("sql_query", {"question": "churn by region"}, "Querying churn"),
        _step_response("DONE", {}, "Analysis complete"),
        '{"findings": ["Churn up in Region 3"], "solutions": [], "recommendation": "Investigate Region 3"}',
    ])
    checkpoints = []
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda step, thought: checkpoints.append((step, thought)),
        max_steps=10,
    )
    assert isinstance(result, AnalystResult)
    assert result.success is True
    assert len(result.steps) == 1  # 1 tool call before DONE


def test_loop_terminates_at_max_steps():
    responses = [
        _step_response("sql_query", {"question": f"query {i}"}, f"Querying {i}")
        for i in range(20)
    ]
    responses.append('{"findings": [], "solutions": [], "recommendation": "Timed out"}')
    agent = _make_agent(responses)
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
        max_steps=3,
    )
    assert isinstance(result, AnalystResult)
    assert len(result.steps) <= 3


def test_max_steps_clamped_to_15():
    responses = [_step_response("DONE")] + ['{"findings": [], "solutions": [], "recommendation": ""}']
    agent = _make_agent(responses)
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
        max_steps=100,
    )
    assert result.success is True


def test_checkpoint_fires_every_3_steps():
    responses = [
        _step_response("sql_query", {"question": f"q{i}"}, f"thought {i}")
        for i in range(6)
    ]
    responses += [
        _step_response("DONE"),
        '{"findings": [], "solutions": [], "recommendation": "done"}',
    ]
    agent = _make_agent(responses)
    checkpoints = []
    agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda step, thought: checkpoints.append(step),
        max_steps=10,
    )
    assert 3 in checkpoints
    assert 6 in checkpoints


def test_analyst_result_has_findings_and_recommendation():
    agent = _make_agent([
        _step_response("DONE"),
        '{"findings": ["Churn up in Region 3"], "solutions": [{"title": "Fix A", "description": "Do A", "pros": ["fast"], "cons": []}], "recommendation": "Fix A"}',
    ])
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
    )
    assert result.findings == ["Churn up in Region 3"]
    assert result.recommendation == "Fix A"
    assert len(result.solutions) == 1


def test_step_records_stored_on_result():
    agent = _make_agent([
        _step_response("sql_query", {"question": "churn"}, "Checking churn"),
        _step_response("DONE"),
        '{"findings": [], "solutions": [], "recommendation": ""}',
    ])
    result = agent.run_deep(
        request=_make_request(),
        config=_make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS),
        on_checkpoint=lambda s, t: None,
    )
    assert len(result.steps) == 1
    assert result.steps[0].tool == "sql_query"
    assert result.steps[0].thought == "Checking churn"
