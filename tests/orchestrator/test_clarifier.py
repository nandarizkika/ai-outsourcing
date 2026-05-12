import pytest
from datetime import datetime
from unittest.mock import MagicMock
from src.orchestrator.clarifier import ClarificationChecker
from src.core.models import Request, Channel, ClarificationState


def make_request(text: str = "show me data") -> Request:
    return Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Budi",
        text=text,
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-1",
    )


@pytest.fixture
def checker():
    llm = MagicMock()
    return ClarificationChecker(llm=llm), llm


def test_clear_request_resolves_immediately(checker):
    c, llm = checker
    llm.complete.return_value = '{"is_clear": true, "questions": []}'
    state = c.check(make_request("show total sales by region for last month"), [])
    assert state.is_resolved is True
    assert state.questions_asked == []


def test_ambiguous_request_generates_questions(checker):
    c, llm = checker
    llm.complete.return_value = '{"is_clear": false, "questions": ["Which time period?", "Which region?"]}'
    state = c.check(make_request("show me the sales data"), [])
    assert state.is_resolved is False
    assert len(state.questions_asked) == 2
    assert state.rounds == 1


def test_exceeding_max_rounds_forces_resolution(checker):
    c, llm = checker
    llm.complete.return_value = '["Assuming last month", "Assuming all regions"]'
    request = make_request("show data")
    existing_state = ClarificationState(
        original_request=request,
        rounds=2,
        questions_asked=["Which period?", "Which region?"],
        answers_received=[],
    )
    state = c.check(request, [], existing_state)
    assert state.is_resolved is True
    assert len(state.assumptions) >= 1


def test_invalid_llm_json_defaults_to_clear(checker):
    c, llm = checker
    llm.complete.return_value = "not valid json"
    state = c.check(make_request("some request"), [])
    assert state.is_resolved is True
