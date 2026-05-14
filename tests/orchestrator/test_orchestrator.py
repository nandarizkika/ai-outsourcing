import pytest
from datetime import datetime
from unittest.mock import MagicMock
from src.orchestrator.orchestrator import Orchestrator
from src.core.models import (
    Request, Channel, ClientConfig, Tier, SkillModule,
    ClarificationState, AgentResult, Response
)
from src.agents.ml_agent import MLAgent
from src.agents.deck_agent import DeckAgent


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


def test_orchestrator_dispatches_ml_agent_when_forecast_request():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": true, "chart": false, "ml": true, "deck": false}',  # _plan
        "Revenue will grow.",  # _generate_response
        "Up 5%.",  # MLAgent._interpret (called inside MLAgent.run)
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock()
    mock_clarifier.check.return_value = MagicMock(is_resolved=True, assumptions=[])
    mock_sql_agent = MagicMock()
    mock_sql_agent.run.return_value = AgentResult(
        agent_name="sql_agent",
        success=True,
        data={"rows": [{"week": i, "revenue": 100 + i * 5} for i in range(10)], "columns": ["week", "revenue"]},
    )
    mock_ml_agent = MagicMock(spec=MLAgent)
    mock_ml_agent.run.return_value = AgentResult(
        agent_name="ml_agent",
        success=True,
        chart_png=b"PNG",
        data={"forecast": [150.0], "target_column": "revenue", "interpretation": "Up 5%."},
    )
    config = MagicMock()
    config.client_id = "client1"
    config.enabled_skills = [SkillModule.SQL_QUERYING, SkillModule.MACHINE_LEARNING]

    orchestrator = Orchestrator(
        llm=mock_llm,
        retriever=mock_retriever,
        clarifier=mock_clarifier,
        sql_agent=mock_sql_agent,
        chart_agent=MagicMock(),
        ml_agent=mock_ml_agent,
    )
    request = Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Alice",
        text="Forecast next 4 weeks of revenue",
        timestamp="2026-05-13T00:00:00",
        client_id="client1",
    )
    result = orchestrator.process(request, config)
    mock_ml_agent.run.assert_called_once()
    assert isinstance(result, Response)
    assert len(result.charts) > 0


def test_orchestrator_dispatches_deck_agent_when_deck_request():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": false, "chart": false, "ml": false, "deck": true}',  # _plan
        "Here is your analysis.",  # _generate_response
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock()
    mock_clarifier.check.return_value = MagicMock(is_resolved=True, assumptions=[])
    mock_deck_agent = MagicMock(spec=DeckAgent)
    mock_deck_agent.run.return_value = AgentResult(
        agent_name="deck_agent", success=True, deck_pptx=b"PPTX"
    )
    config = MagicMock()
    config.client_id = "client1"
    config.enabled_skills = [SkillModule.PRESENTATION_BUILDING]

    orchestrator = Orchestrator(
        llm=mock_llm,
        retriever=mock_retriever,
        clarifier=mock_clarifier,
        sql_agent=MagicMock(),
        chart_agent=MagicMock(),
        deck_agent=mock_deck_agent,
    )
    request = Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Alice",
        text="Create a slide deck for Q1 analysis",
        timestamp="2026-05-13T00:00:00",
        client_id="client1",
    )
    result = orchestrator.process(request, config)
    mock_deck_agent.run.assert_called_once()
    assert isinstance(result, Response)
    assert result.deck_pptx == b"PPTX"
