from src.core.models import (
    Channel, SkillModule, Tier, Request, ClientConfig,
    ClarificationState, AgentResult, Response
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
