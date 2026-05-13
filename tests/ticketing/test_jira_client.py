from unittest.mock import MagicMock, patch
import pytest
from src.ticketing.jira_client import JiraClient
from src.core.models import TicketRef


def _make_client():
    return JiraClient(
        jira_url="https://example.atlassian.net",
        email="test@example.com",
        api_token="tok",
        project_key="AI",
    )


def test_create_ticket_returns_ticket_ref():
    client = _make_client()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"key": "AI-42", "self": "https://example.atlassian.net/rest/api/3/issue/AI-42"}
    with patch("httpx.post", return_value=mock_response) as mock_post:
        ref = client.create_ticket(summary="Analyse churn", description="Full text of the request")
    assert ref.key == "AI-42"
    assert "AI-42" in ref.url
    assert ref.summary == "Analyse churn"
    mock_post.assert_called_once()


def test_long_summary_truncated():
    client = _make_client()
    long_summary = "A" * 300
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"key": "AI-1", "self": "https://example.atlassian.net/rest/api/3/issue/AI-1"}
    with patch("httpx.post", return_value=mock_response) as mock_post:
        ref = client.create_ticket(summary=long_summary, description="desc")
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
    sent_summary = payload["fields"]["summary"]
    assert len(sent_summary) <= 255


def test_http_error_raises():
    client = _make_client()
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 403")
    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(Exception):
            client.create_ticket(summary="x", description="y")
