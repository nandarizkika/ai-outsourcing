import email as email_lib
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch, call
import pytest

from src.channels.email_channel import EmailChannel
from src.core.models import Response, Channel, TicketRef


def _make_channel(ticketing=None):
    mock_orchestrator = MagicMock()
    client_configs = {
        "client1": MagicMock(client_id="client1", enabled_skills=set()),
    }
    return EmailChannel(
        imap_host="imap.example.com",
        imap_user="dian@example.com",
        imap_password="secret",
        smtp_host="smtp.example.com",
        smtp_port=587,
        orchestrator=mock_orchestrator,
        client_configs=client_configs,
        ticketing_service=ticketing,
    ), mock_orchestrator


def _make_raw_email(subject="Analyse revenue", body="Show Q1 revenue", from_addr="user@client1.com"):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = "dian@example.com"
    msg["Message-ID"] = "<test123@example.com>"
    return msg.as_bytes()


def test_poll_processes_unread_email():
    channel, mock_orchestrator = _make_channel()
    mock_orchestrator.process.return_value = Response(
        request_id="r1", text="Q1 revenue is 100M", charts=[]
    )
    raw = _make_raw_email()
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
    mock_orchestrator.process.assert_called_once()
    mock_smtp.sendmail.assert_called_once()


def test_poll_attaches_chart_png():
    channel, mock_orchestrator = _make_channel()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    mock_orchestrator.process.return_value = Response(
        request_id="r1", text="Here is the chart", charts=[png_bytes]
    )
    raw = _make_raw_email()
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
    assert mock_smtp.sendmail.called
    args = mock_smtp.sendmail.call_args[0]
    assert len(args) == 3


def test_poll_no_unread_emails():
    channel, mock_orchestrator = _make_channel()
    with patch("imaplib.IMAP4_SSL") as mock_imap_cls:
        mock_imap = MagicMock()
        mock_imap_cls.return_value.__enter__ = MagicMock(return_value=mock_imap)
        mock_imap_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_imap.search.return_value = ("OK", [b""])
        channel.poll_once(client_id="client1")
    mock_orchestrator.process.assert_not_called()
