# tests/test_integration_phase7.py
import json
from unittest.mock import MagicMock

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.clarifier import ClarificationChecker
from src.agents.report_agent import ReportAgent
from src.core.models import ClientConfig, Tier, SkillModule, Channel, Request


def _make_llm(plan_json=None, response_text="Analysis done."):
    mock = MagicMock()
    def complete(task_type, system, user):
        if "which capabilities" in system.lower():
            return plan_json or '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": false, "ab_test": false, "report": false}'
        if "root-cause" in system.lower() or "deep investigation" in system.lower():
            return "false"
        if "enough information" in system.lower() or "clarif" in system.lower():
            return json.dumps({"resolved": True, "questions": [], "assumptions": []})
        return response_text
    mock.complete.side_effect = complete
    return mock


def _make_config(skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=skills, account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text):
    return Request(
        channel=Channel.SLACK, sender_id="u1", sender_name="User",
        text=text, timestamp="2026-05-17T00:00:00Z", client_id="c1",
    )


def _make_orchestrator(llm, **agents):
    from src.agents.chart_agent import ChartAgent
    from src.agents.sql_agent import SQLAgent
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    clarifier = ClarificationChecker(llm=llm)
    return Orchestrator(
        llm=llm,
        retriever=mock_retriever,
        clarifier=clarifier,
        sql_agent=MagicMock(spec=SQLAgent),
        chart_agent=MagicMock(spec=ChartAgent),
        **agents,
    )


def test_orchestrator_routes_to_report_agent_when_plan_set():
    plan_json = '{"sql": false, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": false, "ab_test": false, "report": true}'
    llm = _make_llm(plan_json=plan_json)
    mock_report = MagicMock(spec=ReportAgent)
    mock_report.run.return_value = MagicMock(
        success=True,
        data={"markdown": "## Summary\n- Good", "html": "<h2>Summary</h2><ul><li>Good</li></ul>"},
    )
    orch = _make_orchestrator(llm, report_agent=mock_report)
    config = _make_config([SkillModule.REPORT_GENERATION])
    result = orch.process(_make_request("Generate a weekly report"), config)
    mock_report.run.assert_called_once()
    assert result.report_markdown == "## Summary\n- Good"
    assert result.report_html == "<h2>Summary</h2><ul><li>Good</li></ul>"


def test_registry_upsert_and_retrieve(tmp_path):
    from src.core.client_registry import ClientRegistry
    reg = ClientRegistry(str(tmp_path / "c.json"))
    config = _make_config([SkillModule.SQL_QUERYING])
    reg.upsert(config)
    result = reg.get("c1")
    assert result is not None
    assert result.client_id == "c1"


def test_orchestrator_uses_registry_connector_over_sql_agent():
    from sqlalchemy import create_engine, text as sa_text
    from src.connectors.sql import SQLConnector

    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(sa_text("CREATE TABLE metrics (month TEXT, revenue INTEGER)"))
        conn.execute(sa_text("INSERT INTO metrics VALUES ('Jan', 1000)"))
        conn.commit()
    connector = SQLConnector(connection_url="sqlite:///:memory:", engine=engine)

    mock_registry = MagicMock()
    mock_registry.get_connector.return_value = connector

    plan_json = '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": false, "ab_test": false, "report": false}'

    def complete(task_type, system, user):
        if "which capabilities" in system.lower():
            return plan_json
        if "root-cause" in system.lower() or "deep investigation" in system.lower():
            return "false"
        if "enough information" in system.lower() or "clarif" in system.lower():
            return json.dumps({"resolved": True, "questions": [], "assumptions": []})
        if "sql expert" in system.lower():
            return "SELECT * FROM metrics"
        return "Analysis done."

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = complete

    orch = _make_orchestrator(mock_llm, registry=mock_registry)
    config = _make_config([SkillModule.SQL_QUERYING])
    result = orch.process(_make_request("Show me metrics"), config)

    mock_registry.get_connector.assert_called_once_with("c1")
    assert hasattr(result, "text")
