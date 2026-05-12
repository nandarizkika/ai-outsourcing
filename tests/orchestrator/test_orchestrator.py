import pytest
from datetime import datetime
from unittest.mock import MagicMock
from src.orchestrator.orchestrator import Orchestrator
from src.core.models import (
    Request, Channel, ClientConfig, Tier, SkillModule,
    ClarificationState, AgentResult, Response
)


def make_request(text: str = "show total sales by region") -> Request:
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Budi",
        text=text, timestamp=datetime.utcnow().isoformat(), client_id="client-1",
    )


def make_config(tier: Tier = Tier.ADVANCED) -> ClientConfig:
    from src.skill_modules import SkillModuleRegistry
    return ClientConfig(
        client_id="client-1", name="Test Corp", tier=tier,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(tier),
        account_mode="vendor", active_channels=[Channel.SLACK],
    )


@pytest.fixture
def orchestrator():
    llm = MagicMock()
    retriever = MagicMock()
    clarifier = MagicMock()
    sql_agent = MagicMock()
    chart_agent = MagicMock()
    return Orchestrator(llm, retriever, clarifier, sql_agent, chart_agent), \
           llm, retriever, clarifier, sql_agent, chart_agent


def test_returns_clarification_state_when_unclear(orchestrator):
    orch, llm, retriever, clarifier, sql_agent, chart_agent = orchestrator
    retriever.search.return_value = []
    clarifier.check.return_value = ClarificationState(
        original_request=make_request(), rounds=1,
        questions_asked=["Which time period?"], is_resolved=False,
    )

    result = orch.process(make_request("show data"), make_config())
    assert isinstance(result, ClarificationState)
    sql_agent.run.assert_not_called()


def test_returns_response_when_clear(orchestrator):
    orch, llm, retriever, clarifier, sql_agent, chart_agent = orchestrator
    retriever.search.return_value = ["Sales table contains region and amount"]
    clarifier.check.return_value = ClarificationState(
        original_request=make_request(), is_resolved=True,
    )
    llm.complete.side_effect = [
        '{"sql": true, "chart": true}',   # plan
        "Here is the sales analysis...",   # response text
    ]
    sql_agent.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT 1", "rows": [{"region": "Jakarta", "total": 1000}], "columns": ["region", "total"]}
    )
    chart_agent.run.return_value = AgentResult(
        agent_name="chart_agent", success=True, chart_png=b"\x89PNG..."
    )

    result = orch.process(make_request(), make_config())
    assert isinstance(result, Response)
    assert result.text == "Here is the sales analysis..."
    assert len(result.charts) == 1


def test_sql_skill_disabled_skips_sql_agent(orchestrator):
    orch, llm, retriever, clarifier, sql_agent, chart_agent = orchestrator
    retriever.search.return_value = []
    clarifier.check.return_value = ClarificationState(
        original_request=make_request(), is_resolved=True,
    )
    llm.complete.side_effect = [
        '{"sql": true, "chart": true}',
        "I cannot run SQL queries for your current plan.",
    ]
    config = ClientConfig(
        client_id="client-1", name="Test", tier=Tier.BASIC,
        enabled_skills=[SkillModule.REPORT_GENERATION],  # no SQL skill
        account_mode="vendor", active_channels=[Channel.SLACK],
    )

    result = orch.process(make_request(), config)
    assert isinstance(result, Response)
    sql_agent.run.assert_not_called()
