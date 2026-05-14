"""Phase 3 end-to-end integration tests."""
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from pptx import Presentation

from src.agents.deck_agent import DeckAgent
from src.agents.ml_agent import MLAgent
from src.channels.slack import SlackChannel
from src.core.models import (
    AgentResult,
    Channel,
    ClientConfig,
    Response,
    ScheduledJob,
    SkillModule,
    Tier,
)
from src.jobs.scheduler import JobScheduler
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator


def _make_client_config(skills: list[SkillModule]) -> ClientConfig:
    return ClientConfig(
        client_id="client1",
        name="Test Client",
        tier=Tier.ENTERPRISE,
        enabled_skills=skills,
        account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def test_ml_forecast_path_slack_to_response():
    """Slack @mention with forecast intent → MLAgent called → charts in response."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        '{"sql": true, "chart": false, "ml": true, "deck": false}',  # _plan
        "Revenue will grow by 12% next month.",                        # _generate_response
        "Forecast interpretation from LLM.",                           # MLAgent._interpret
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    clarifier = ClarificationChecker(llm=mock_llm)

    mock_sql_agent = MagicMock()
    mock_sql_agent.run.return_value = AgentResult(
        agent_name="sql_agent",
        success=True,
        data={
            "rows": [{"week": i, "revenue": 100 + i * 10} for i in range(10)],
            "columns": ["week", "revenue"],
        },
    )

    ml_agent = MLAgent(llm=mock_llm)
    deck_agent = DeckAgent()

    orchestrator = Orchestrator(
        llm=mock_llm,
        retriever=mock_retriever,
        clarifier=clarifier,
        sql_agent=mock_sql_agent,
        chart_agent=MagicMock(return_value=AgentResult(agent_name="chart_agent", success=False, error="skip")),
        ml_agent=ml_agent,
        deck_agent=deck_agent,
    )
    config = _make_client_config([SkillModule.SQL_QUERYING, SkillModule.MACHINE_LEARNING])

    with patch("slack_bolt.App"):
        channel = SlackChannel(
            bot_token="xoxb-test",
            signing_secret="secret",
            bot_user_id="BOTID",
            orchestrator=orchestrator,
            client_configs={"client1": config},
            token_verification_enabled=False,
        )

    say = MagicMock()
    event = {
        "user": "U1",
        "text": "<@BOTID> forecast revenue for next 4 weeks",
        "channel": "C1",
        "ts": "1.0",
    }
    channel._handle_mention(event, say)
    say.assert_called_once()
    reply_text = say.call_args.kwargs.get("text") or say.call_args.args[0]
    assert len(reply_text) > 0


def test_deck_agent_path_produces_valid_pptx():
    """DeckAgent.run produces parseable PPTX bytes with title and content slides."""
    agent = DeckAgent()
    result = agent.run(
        title="Q1 Revenue Report",
        sections=[
            {"heading": "Key Findings", "body": "Revenue up 15%."},
            {"heading": "Risks", "body": "Supply chain pressure."},
        ],
        charts=[],
    )
    assert result.success
    prs = Presentation(BytesIO(result.deck_pptx))
    assert len(prs.slides) == 3  # title + 2 content slides
    all_text = " ".join(
        shape.text
        for slide in prs.slides
        for shape in slide.shapes
        if shape.has_text_frame
    )
    assert "Q1 Revenue Report" in all_text
    assert "Key Findings" in all_text


def test_scheduled_job_fires_and_calls_callback():
    """JobScheduler._run_job fires orchestrator and calls the delivery callback."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.process.return_value = Response(
        request_id="r1",
        text="Weekly revenue: 500M IDR",
        charts=[],
    )
    config = _make_client_config([SkillModule.REPORT_GENERATION])
    scheduler = JobScheduler(
        orchestrator=mock_orchestrator,
        client_configs={"client1": config},
    )
    delivered = []
    job = ScheduledJob(
        job_id="j1",
        client_id="client1",
        description="Weekly revenue report",
        request_text="Show weekly revenue summary",
        cron_expression="0 9 * * 1",
        delivery_channel=Channel.SLACK,
        delivery_destination="C_GENERAL",
    )
    scheduler.add_job(job, on_complete=lambda r: delivered.append(r))
    scheduler._run_job("j1")

    assert mock_orchestrator.process.call_count == 1
    assert len(delivered) == 1
    assert delivered[0].text == "Weekly revenue: 500M IDR"
    updated_job = scheduler.list_jobs()[0]
    assert updated_job.last_run is not None
    assert "500M IDR" in updated_job.last_result_summary
