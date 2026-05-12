from datetime import datetime
from unittest.mock import MagicMock
from src.channels.slack import SlackChannel
from src.core.models import (
    Request, Channel, ClientConfig, Tier, ClarificationState, Response
)
from src.skill_modules import SkillModuleRegistry


def make_config() -> ClientConfig:
    return ClientConfig(
        client_id="client-1", name="Test Corp", tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor", active_channels=[Channel.SLACK],
    )


def make_slack_event(text: str = "show sales data", thread_ts: str = "123.456") -> dict:
    return {
        "user": "U123",
        "text": f"<@BOTID> {text}",
        "ts": thread_ts,
        "channel": "C001",
        "team": "T001",
    }


def make_request() -> Request:
    return Request(
        channel=Channel.SLACK,
        sender_id="U123",
        sender_name="U123",
        text="test",
        thread_id="123.456",
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-1",
    )


def test_normalizes_mention_to_request():
    orchestrator = MagicMock()
    orchestrator.process.return_value = Response(request_id="r1", text="Here is the data.")

    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
    )

    request = channel._build_request(make_slack_event(), "T001")

    assert isinstance(request, Request)
    assert request.channel == Channel.SLACK
    assert request.client_id == "client-1"
    assert "show sales data" in request.text
    assert "<@BOTID>" not in request.text


def test_returns_clarification_questions_as_message():
    orchestrator = MagicMock()
    orchestrator.process.return_value = ClarificationState(
        original_request=make_request(),
        rounds=1,
        questions_asked=["Which time period?", "Which region?"],
        is_resolved=False,
    )

    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
    )
    say = MagicMock()
    channel._handle_result(
        result=orchestrator.process.return_value,
        event=make_slack_event(),
        say=say,
    )

    say.assert_called_once()
    call_text = say.call_args[1]["text"]
    assert "Which time period?" in call_text


def test_unknown_workspace_sends_error():
    orchestrator = MagicMock()

    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={},  # no config for this workspace
    )
    say = MagicMock()
    channel._on_unconfigured_workspace(say=say, thread_ts="123.456")
    say.assert_called_once()
