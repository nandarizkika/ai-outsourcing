from unittest.mock import MagicMock
import pytest
from src.ticketing.ticketing_service import TicketingService
from src.core.models import Request, Channel, TicketRef


def _make_service(ticket_ref=None):
    mock_client = MagicMock()
    if ticket_ref is None:
        ticket_ref = TicketRef(key="AI-1", url="https://jira.example.com/browse/AI-1", summary="x")
    mock_client.create_ticket.return_value = ticket_ref
    return TicketingService(jira_client=mock_client), mock_client


def test_creates_ticket_from_request():
    service, mock_client = _make_service()
    request = Request(
        client_id="client1",
        sender_id="user1",
        sender_name="Alice",
        text="Show me monthly revenue for Q1",
        channel=Channel.SLACK,
        timestamp="2026-05-13T07:00:00Z",
    )
    ref = service.create_for_request(request)
    assert ref.key == "AI-1"
    mock_client.create_ticket.assert_called_once()
    call_kwargs = mock_client.create_ticket.call_args
    summary = call_kwargs.kwargs.get("summary") or call_kwargs.args[0]
    assert "monthly revenue" in summary.lower() or "Show me" in summary


def test_truncates_long_request_text():
    service, mock_client = _make_service()
    long_text = "Analyse " + "x" * 300
    request = Request(
        client_id="client1",
        sender_id="user2",
        sender_name="Bob",
        text=long_text,
        channel=Channel.EMAIL,
        timestamp="2026-05-13T07:00:00Z",
    )
    service.create_for_request(request)
    call_kwargs = mock_client.create_ticket.call_args
    summary = call_kwargs.kwargs.get("summary") or call_kwargs.args[0]
    assert len(summary) <= 255
