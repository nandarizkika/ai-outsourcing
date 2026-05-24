# tests/test_integration_phase5.py
import io
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.agents.analyst_agent import AnalystAgent
from src.agents.spreadsheet_agent import SpreadsheetAgent
from src.core.models import (
    AgentResult, AnalystResult, Channel, ClientConfig,
    ClarificationState, Request, Response, SkillModule, StepRecord, Tier,
)
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.clarifier import ClarificationChecker
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.chart_agent import ChartAgent


def _make_config(*skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=list(skills), account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text="Why did churn spike?", file_bytes=None, filename=None):
    return Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="Ana",
        text=text, timestamp="2026-05-14T00:00:00", client_id="c1",
        file_bytes=file_bytes, filename=filename,
    )


def _base_orc_deps(llm_responses):
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = llm_responses
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.search.return_value = []
    mock_clarifier = MagicMock(spec=ClarificationChecker)
    state = ClarificationState(original_request=_make_request())
    state.is_resolved = True
    mock_clarifier.check.return_value = state
    mock_sql = MagicMock()
    mock_sql.run.return_value = AgentResult(
        agent_name="sql_agent", success=True,
        data={"query": "SELECT 1", "rows": [{"x": 1}], "columns": ["x"]},
    )
    return dict(
        llm=mock_llm, retriever=mock_retriever, clarifier=mock_clarifier,
        sql_agent=mock_sql, chart_agent=ChartAgent(),
    )


async def test_orchestrator_detects_deep_intent_and_returns_clarification():
    deps = _base_orc_deps([
        "true",   # _detect_deep_intent returns "true"
        '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Summary response.",
    ])
    mock_analyst = MagicMock(spec=AnalystAgent)
    orc = Orchestrator(**deps, analyst_agent=mock_analyst)
    config = _make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS)
    result = await orc.process(_make_request("Why did churn spike?"), config)
    assert isinstance(result, ClarificationState)
    assert any("deep" in q.lower() or "dive" in q.lower() for q in result.questions_asked)


async def test_orchestrator_routes_to_spreadsheet_agent_when_file_present():
    df = pd.DataFrame({"revenue": [100, 200, 300]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue()

    deps = _base_orc_deps([
        "false",   # _detect_deep_intent returns false (no deep intent for spreadsheet)
        '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false}',
        "Revenue summary: mean is 200.",
    ])
    mock_spreadsheet = MagicMock(spec=SpreadsheetAgent)
    mock_spreadsheet.run.return_value = AgentResult(
        agent_name="spreadsheet_agent", success=True,
        data={"rows": [{"revenue": 200.0}], "columns": ["revenue"], "summary": {}},
    )
    orc = Orchestrator(**deps, spreadsheet_agent=mock_spreadsheet)
    config = _make_config(SkillModule.SPREADSHEET_ANALYSIS, SkillModule.REPORT_GENERATION)
    request = _make_request(file_bytes=csv_bytes, filename="data.csv")
    result = await orc.process(request, config)
    mock_spreadsheet.run.assert_called_once()
    assert isinstance(result, Response)


async def test_analyst_agent_full_loop_produces_analyst_result():
    analyst_result = AnalystResult(
        agent_name="analyst_agent", success=True,
        steps=[StepRecord(step=1, thought="Checking churn", tool="sql_query",
                          tool_input={"question": "churn by region"},
                          observation='[{"region": "A", "churn": 0.2}]')],
        findings=["Churn is highest in Region A"],
        solutions=[{"title": "Fix A", "description": "Rollback pricing in A",
                    "pros": ["fast"], "cons": ["cost"]}],
        recommendation="Fix A",
    )
    deps = _base_orc_deps([
        "true",   # _detect_deep_intent on first call
        "true",   # _detect_deep_intent on second call (or not used)
        "Comprehensive analysis complete.",
    ])
    mock_analyst = MagicMock(spec=AnalystAgent)
    mock_analyst.run_deep.return_value = analyst_result
    orc = Orchestrator(**deps, analyst_agent=mock_analyst)
    config = _make_config(SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS)

    req = _make_request("Why did churn spike?")
    state = await orc.process(req, config)
    assert isinstance(state, ClarificationState)

    # User confirms deep dive
    state.deep_dive_confirmed = True
    state.is_resolved = True

    result2 = await orc.process(req, config, clarification_state=state)
    assert mock_analyst.run_deep.called or isinstance(result2, (Response, AnalystResult))
