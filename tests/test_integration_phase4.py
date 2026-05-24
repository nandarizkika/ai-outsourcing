# tests/test_integration_phase4.py
import json
import pytest
from unittest.mock import MagicMock

from src.agents.anomaly_agent import AnomalyAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.knowledge.retriever import KnowledgeRetriever
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.agents.chart_agent import ChartAgent
from src.core.models import (
    AgentResult, Channel, ClarificationState, ClientConfig,
    Request, Response, SkillModule, Tier,
)


def _make_request(text="show churn", client_id="c1"):
    return Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Ana",
        text=text,
        timestamp="2026-05-14T00:00:00",
        client_id=client_id,
    )


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test Corp", tier=Tier.ADVANCED,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_orchestrator(anomaly_agent=None, memory_logger=None, retriever_search=None):
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Churn rate is elevated at 15%.",
    ]
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = retriever_search if retriever_search is not None else []
    mock_clarifier = MagicMock(spec=ClarificationChecker)
    state = ClarificationState(original_request=_make_request())
    state.is_resolved = True
    mock_clarifier.check.return_value = state
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT churn FROM metrics",
              "rows": [{"churn": 0.15}], "columns": ["churn"]},
    )
    return Orchestrator(
        llm=mock_llm,
        retriever=mock_retriever,
        clarifier=mock_clarifier,
        sql_agent=mock_sql,
        chart_agent=ChartAgent(),
        anomaly_agent=anomaly_agent,
        memory_logger=memory_logger,
    )


async def test_anomaly_agent_flags_critical_breach_in_response():
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = [
        json.dumps({"metric": "churn", "operator": ">", "threshold": 0.10, "severity": "critical"})
    ]
    anomaly_agent = AnomalyAgent(retriever=mock_retriever)
    orc = _make_orchestrator(anomaly_agent=anomaly_agent)
    config = _make_config(SkillModule.SQL_QUERYING, SkillModule.HARD_RULE_ANOMALY)
    result = await orc.process(_make_request(), config)
    assert isinstance(result, Response)
    assert len(result.anomalies) == 1
    assert result.anomalies[0].severity == "critical"
    assert result.anomalies[0].metric == "churn"


async def test_interaction_memory_logger_receives_log_call():
    mock_logger = MagicMock(spec=InteractionMemoryLogger)
    orc = _make_orchestrator(memory_logger=mock_logger)
    config = _make_config(SkillModule.SQL_QUERYING)
    await orc.process(_make_request(text="show revenue"), config)
    mock_logger.log.assert_called_once()
    call_kwargs = mock_logger.log.call_args[1]
    assert call_kwargs["request"].text == "show revenue"
    assert "SELECT churn FROM metrics" in call_kwargs["sql_queries"]


async def test_no_anomalies_when_skill_not_in_config():
    mock_anomaly = MagicMock(spec=AnomalyAgent)
    orc = _make_orchestrator(anomaly_agent=mock_anomaly)
    config = _make_config(SkillModule.SQL_QUERYING)
    result = await orc.process(_make_request(), config)
    mock_anomaly.run.assert_not_called()
    assert result.anomalies == []
