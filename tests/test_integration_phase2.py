"""Phase 2 end-to-end integration tests.

Full path: channel event → TicketingService (real) → JiraClient (mocked)
→ ticket key appears in the channel reply.
"""

from email.mime.text import MIMEText
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.channels.email_channel import EmailChannel
from src.channels.slack import SlackChannel
from src.core.models import (
    Channel,
    ClientConfig,
    Response,
    SkillModule,
    Tier,
    TicketRef,
)
from src.ticketing.jira_client import JiraClient
from src.ticketing.ticketing_service import TicketingService
from src.skill_modules import SkillModuleRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client_config(client_id: str = "client1") -> ClientConfig:
    """Build a real ClientConfig with sensible defaults."""
    return ClientConfig(
        client_id=client_id,
        name="Test Corp",
        tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor",
        active_channels=[Channel.SLACK, Channel.EMAIL],
    )


# ---------------------------------------------------------------------------
# Slack end-to-end
# ---------------------------------------------------------------------------


def test_slack_end_to_end_with_ticketing():
    """Full path: Slack mention → TicketingService creates ticket → reply contains ticket key."""
    # Mock JiraClient to return a real TicketRef
    mock_jira_client = MagicMock(spec=JiraClient)
    mock_jira_client.create_ticket.return_value = TicketRef(
        key="AI-123",
        url="https://example.atlassian.net/browse/AI-123",
        summary="show me monthly revenue",
    )

    # Real TicketingService wrapping the mock client
    ticketing = TicketingService(jira_client=mock_jira_client)

    # Mock orchestrator returns a clean response
    mock_orchestrator = MagicMock()
    mock_orchestrator.process = AsyncMock(return_value=Response(
        request_id="r1",
        text="Revenue for Q1 is 500M",
        charts=[],
    ))

    # Real ClientConfig — _handle_mention looks up by team ID ("T001")
    config = _make_client_config("client1")

    with patch("slack_bolt.App"):
        channel = SlackChannel(
            bot_token="xoxb-test",
            signing_secret="secret",
            bot_user_id="BOTID",
            orchestrator=mock_orchestrator,
            client_configs={"T001": config},
            token_verification_enabled=False,
            ticketing_service=ticketing,
        )

    say = MagicMock()
    event = {
        "user": "U123",
        "text": "<@BOTID> show me monthly revenue",
        "channel": "C1",
        "ts": "1234567890.000",
        "team": "T001",
    }
    channel._handle_mention(event, say)

    # Verify ticket was created via real TicketingService → mock JiraClient
    mock_jira_client.create_ticket.assert_called_once()

    # Verify ticket key appears in the reply
    say.assert_called_once()
    reply_text = say.call_args.kwargs.get("text") or say.call_args.args[0]
    assert "AI-123" in reply_text
    assert "Revenue for Q1 is 500M" in reply_text


# ---------------------------------------------------------------------------
# Email end-to-end
# ---------------------------------------------------------------------------


def test_email_end_to_end_with_ticketing():
    """Full path: Email poll → TicketingService creates ticket → SMTP reply contains ticket key."""
    mock_jira_client = MagicMock(spec=JiraClient)
    mock_jira_client.create_ticket.return_value = TicketRef(
        key="AI-456",
        url="https://example.atlassian.net/browse/AI-456",
        summary="analyse churn",
    )
    ticketing = TicketingService(jira_client=mock_jira_client)

    mock_orchestrator = MagicMock()
    mock_orchestrator.process = AsyncMock(return_value=Response(
        request_id="r2",
        text="Churn rate is 5%",
        charts=[],
    ))

    # EmailChannel uses client_configs as a plain dict — MagicMock is fine here
    # because poll_once only calls client_configs.get(client_id) and passes the
    # result straight through to orchestrator.process without inspecting fields.
    client_config = MagicMock()
    client_config.client_id = "client1"

    channel = EmailChannel(
        imap_host="imap.example.com",
        imap_user="dian@example.com",
        imap_password="secret",
        smtp_host="smtp.example.com",
        smtp_port=587,
        orchestrator=mock_orchestrator,
        client_configs={"client1": client_config},
        ticketing_service=ticketing,
    )

    msg = MIMEText("analyse churn rate for last month")
    msg["Subject"] = "Analyse churn"
    msg["From"] = "manager@client1.com"
    msg["To"] = "dian@example.com"
    msg["Message-ID"] = "<abc@example.com>"
    raw = msg.as_bytes()

    with patch("imaplib.IMAP4_SSL") as mock_imap_cls, \
         patch("smtplib.SMTP") as mock_smtp_cls:
        mock_imap = MagicMock()
        mock_imap_cls.return_value.__enter__ = MagicMock(return_value=mock_imap)
        mock_imap_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_imap.search.return_value = ("OK", [b"1"])
        mock_imap.fetch.return_value = ("OK", [(b"1 (RFC822 {100})", raw)])

        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        channel.poll_once(client_id="client1")

    # Verify real TicketingService called mock JiraClient
    mock_jira_client.create_ticket.assert_called_once()

    # Verify ticket key appears in the SMTP reply
    mock_smtp.sendmail.assert_called_once()
    sent_bytes = mock_smtp.sendmail.call_args[0][2]
    assert b"AI-456" in sent_bytes
