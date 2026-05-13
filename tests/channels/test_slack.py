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


def _make_channel(ticketing=None):
    orchestrator = MagicMock()
    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
        token_verification_enabled=False,
        ticketing_service=ticketing,
    )
    return channel, orchestrator


def test_normalizes_mention_to_request():
    orchestrator = MagicMock()
    orchestrator.process.return_value = Response(request_id="r1", text="Here is the data.")

    channel = SlackChannel(
        bot_token="xoxb-fake",
        signing_secret="fake-secret",
        bot_user_id="BOTID",
        orchestrator=orchestrator,
        client_configs={"T001": make_config()},
        token_verification_enabled=False,
    )

    request = channel._build_request(make_slack_event(), make_config())

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
        token_verification_enabled=False,
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
        token_verification_enabled=False,
    )
    say = MagicMock()
    channel._on_unconfigured_workspace(say=say, thread_ts="123.456")
    say.assert_called_once()


def test_ticketing_service_called_on_mention():
    mock_ticketing = MagicMock()
    mock_ticketing.create_for_request.return_value = MagicMock(key="AI-7", url="...", summary="x")
    channel, mock_orchestrator = _make_channel(ticketing=mock_ticketing)
    mock_orchestrator.process.return_value = Response(request_id="r1", text="done", charts=[])
    event = {"user": "U123", "text": "<@BOTID> show me revenue", "channel": "C1", "ts": "1.0", "team": "T001"}
    say = MagicMock()
    channel._handle_mention(event, say)
    mock_ticketing.create_for_request.assert_called_once()


def test_ticket_key_prepended_to_reply():
    mock_ticketing = MagicMock()
    mock_ticketing.create_for_request.return_value = MagicMock(key="AI-99", url="...", summary="x")
    channel, mock_orchestrator = _make_channel(ticketing=mock_ticketing)
    mock_orchestrator.process.return_value = Response(request_id="r1", text="Analysis done", charts=[])
    event = {"user": "U123", "text": "<@BOTID> analyse churn", "channel": "C1", "ts": "1.0", "team": "T001"}
    say = MagicMock()
    channel._handle_mention(event, say)
    say_call = say.call_args
    text_sent = say_call.kwargs.get("text") or say_call.args[0]
    assert "AI-99" in text_sent
