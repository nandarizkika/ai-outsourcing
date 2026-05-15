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
from src.agents.anomaly_agent import AnomalyAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.orchestrator.clarifier import ClarificationChecker
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.chart_agent import ChartAgent


def _make_orc_deps():
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Analysis complete.",
    ]
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock(spec=ClarificationChecker)
    state = ClarificationState(original_request=_make_request())
    state.is_resolved = True
    mock_clarifier.check.return_value = state
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT 1", "rows": [{"churn": 0.15}], "columns": ["churn"]},
    )
    return dict(
        llm=mock_llm, retriever=mock_retriever, clarifier=mock_clarifier,
        sql_agent=mock_sql, chart_agent=ChartAgent(),
    )


def _make_request(text="show churn"):
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Ana",
        text=text, timestamp="2026-05-14T00:00:00", client_id="c1",
    )


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ADVANCED,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def test_orchestrator_runs_anomaly_agent_when_skill_enabled():
    deps = _make_orc_deps()
    mock_anomaly = MagicMock()
    mock_anomaly.run.return_value = AgentResult(
        agent_name="anomaly_agent", success=True,
        data={"anomalies": [{"metric": "churn", "value": 0.15, "severity": "critical",
                              "description": "churn breach", "mode": "hard_rule",
                              "expected": None, "operator": ">", "threshold": 0.1}]},
    )
    orc = Orchestrator(**deps, anomaly_agent=mock_anomaly)
    result = orc.process(_make_request(), _make_config(SkillModule.SQL_QUERYING, SkillModule.HARD_RULE_ANOMALY))
    mock_anomaly.run.assert_called_once()
    assert isinstance(result, Response)
    assert len(result.anomalies) == 1


def test_orchestrator_skips_anomaly_when_skill_not_enabled():
    deps = _make_orc_deps()
    mock_anomaly = MagicMock()
    orc = Orchestrator(**deps, anomaly_agent=mock_anomaly)
    orc.process(_make_request(), _make_config(SkillModule.SQL_QUERYING))
    mock_anomaly.run.assert_not_called()


def test_orchestrator_calls_memory_logger():
    deps = _make_orc_deps()
    mock_logger = MagicMock()
    orc = Orchestrator(**deps, memory_logger=mock_logger)
    result = orc.process(_make_request(text="show revenue"), _make_config(SkillModule.SQL_QUERYING))
    mock_logger.log.assert_called_once()
    call_kwargs = mock_logger.log.call_args[1]
    assert call_kwargs["request"].text == "show revenue"


def test_orchestrator_funnel_mode_passes_analysis_mode_to_sql():
    deps = _make_orc_deps()
    deps["llm"].complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": true, "cohort": false}',
        "Funnel analysis complete.",
    ]
    # retriever returns funnel docs so we don't get a clarification
    deps["retriever"].search.return_value = ["Visit -> Signup -> Active -> Paid"]
    orc = Orchestrator(**deps)
    orc.process(_make_request(text="show funnel"), _make_config(SkillModule.SQL_QUERYING, SkillModule.FUNNEL_ANALYSIS))
    call_args = deps["sql_agent"].run.call_args
    assert call_args[1].get("analysis_mode") == "funnel"


def test_funnel_returns_clarification_when_no_kb_definition():
    deps = _make_orc_deps()
    def search_side_effect(client_id, query):
        if "funnel" in query.lower():
            return []
        return []  # general context also empty in this test
    deps["retriever"].search.side_effect = search_side_effect
    deps["llm"].complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": true, "cohort": false}',
    ]
    orc = Orchestrator(**deps)
    result = orc.process(
        _make_request(text="show funnel"),
        _make_config(SkillModule.SQL_QUERYING, SkillModule.FUNNEL_ANALYSIS),
    )
    assert isinstance(result, ClarificationState)
    assert "funnel" in result.questions_asked[0].lower()


def test_funnel_injects_kb_definition_into_context():
    deps = _make_orc_deps()
    # retriever.search returns funnel definition on second call (first call is general context)
    call_count = {"n": 0}
    def search_side_effect(client_id, query):
        call_count["n"] += 1
        if "funnel" in query.lower():
            return ["Visit -> Signup -> Active -> Paid"]
        return []
    deps["retriever"].search.side_effect = search_side_effect
    deps["llm"].complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": true, "cohort": false}',
        "Funnel analysis complete.",
    ]
    orc = Orchestrator(**deps)
    result = orc.process(
        _make_request(text="show funnel"),
        _make_config(SkillModule.SQL_QUERYING, SkillModule.FUNNEL_ANALYSIS),
    )
    assert isinstance(result, Response)
    # SQL agent should have been called with context that includes funnel definition
    call_args = deps["sql_agent"].run.call_args
    sql_context = call_args[0][2]  # positional arg 3
    assert any("Visit" in c for c in sql_context)


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
