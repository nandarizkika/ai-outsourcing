from src.core.models import (
    Channel, SkillModule, Tier, Request, ClientConfig,
    ClarificationState, AgentResult, Response, TicketRef
)
from datetime import datetime

def test_request_model():
    r = Request(
        channel=Channel.SLACK,
        sender_id="U123",
        sender_name="Budi",
        text="show me churn rate",
        thread_id="1234567890.000100",
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-abc",
    )
    assert r.channel == Channel.SLACK
    assert r.client_id == "client-abc"

def test_client_config_basic_tier():
    config = ClientConfig(
        client_id="client-abc",
        name="Test Corp",
        tier=Tier.BASIC,
        enabled_skills=[
            SkillModule.SQL_QUERYING,
            SkillModule.DATA_VISUALIZATION,
            SkillModule.REPORT_GENERATION,
            SkillModule.SCHEDULED_REPORTING,
            SkillModule.HARD_RULE_ANOMALY,
        ],
        account_mode="vendor",
        active_channels=[Channel.SLACK],
    )
    assert SkillModule.SQL_QUERYING in config.enabled_skills
    assert SkillModule.MACHINE_LEARNING not in config.enabled_skills

def test_clarification_state_defaults():
    r = Request(
        channel=Channel.SLACK, sender_id="U1", sender_name="A",
        text="show data", timestamp="2026-01-01T00:00:00", client_id="c1"
    )
    state = ClarificationState(original_request=r)
    assert state.rounds == 0
    assert state.is_resolved is False
    assert state.assumptions == []

def test_agent_result_success():
    result = AgentResult(
        agent_name="sql_agent",
        success=True,
        data={"query": "SELECT 1", "rows": [{"count": 1}], "columns": ["count"]}
    )
    assert result.success is True
    assert result.error is None

def test_response_defaults():
    resp = Response(request_id="req-1", text="Here is your analysis.")
    assert resp.charts == []
    assert resp.assumptions == []


def test_ticket_ref_fields():
    ref = TicketRef(key="AI-42", url="https://jira.example.com/browse/AI-42", summary="Analyse churn")
    assert ref.key == "AI-42"
    assert ref.url == "https://jira.example.com/browse/AI-42"
    assert ref.summary == "Analyse churn"


def test_response_ticket_defaults_none():
    r = Response(request_id="r1", text="ok")
    assert r.ticket is None


def test_response_ticket_accepts_ticket_ref():
    ref = TicketRef(key="AI-1", url="https://jira.example.com/browse/AI-1", summary="x")
    r = Response(request_id="r1", text="ok", ticket=ref)
    assert r.ticket.key == "AI-1"
