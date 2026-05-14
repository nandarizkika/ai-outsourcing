from src.core.models import (
    Channel, SkillModule, Tier, Request, ClientConfig,
    ClarificationState, AgentResult, Response, TicketRef, ScheduledJob, Anomaly
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


def test_scheduled_job_fields():
    job = ScheduledJob(
        job_id="j1",
        client_id="client1",
        description="Weekly revenue report",
        request_text="Show me weekly revenue",
        cron_expression="0 9 * * 1",
        delivery_channel=Channel.SLACK,
        delivery_destination="C_GENERAL",
    )
    assert job.job_id == "j1"
    assert job.cron_expression == "0 9 * * 1"
    assert job.last_run is None


def test_agent_result_deck_pptx_defaults_none():
    r = AgentResult(agent_name="deck_agent", success=True)
    assert r.deck_pptx is None


def test_response_deck_pptx_defaults_none():
    r = Response(request_id="r1", text="ok")
    assert r.deck_pptx is None


def test_response_deck_pptx_accepts_bytes():
    r = Response(request_id="r1", text="ok", deck_pptx=b"PPTX_DATA")
    assert r.deck_pptx == b"PPTX_DATA"


def test_anomaly_model_fields():
    a = Anomaly(
        metric="churn_rate",
        value=0.15,
        threshold=0.10,
        operator=">",
        severity="critical",
        description="churn_rate is 0.15 (rule: churn_rate > 0.1)",
        mode="hard_rule",
    )
    assert a.metric == "churn_rate"
    assert a.severity == "critical"
    assert a.mode == "hard_rule"


def test_anomaly_defaults():
    a = Anomaly(metric="revenue", value=500.0, description="outlier", mode="statistical")
    assert a.severity == "warning"
    assert a.expected is None
    assert a.threshold is None
    assert a.operator is None


def test_response_anomalies_defaults_empty():
    r = Response(request_id="r1", text="ok")
    assert r.anomalies == []


def test_response_accepts_anomalies():
    a = Anomaly(metric="x", value=1.0, description="test", mode="hard_rule")
    r = Response(request_id="r1", text="ok", anomalies=[a])
    assert len(r.anomalies) == 1
    assert r.anomalies[0].metric == "x"


def test_skill_module_has_funnel_and_cohort():
    assert SkillModule.FUNNEL_ANALYSIS == "funnel_analysis"
    assert SkillModule.COHORT_ANALYSIS == "cohort_analysis"
