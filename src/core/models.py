from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel


class Channel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    JIRA = "jira"


class SkillModule(str, Enum):
    SQL_QUERYING = "sql_querying"
    DATA_VISUALIZATION = "data_visualization"
    REPORT_GENERATION = "report_generation"
    SCHEDULED_REPORTING = "scheduled_reporting"
    HARD_RULE_ANOMALY = "hard_rule_anomaly"
    SPREADSHEET_ANALYSIS = "spreadsheet_analysis"
    LOOKER_INTEGRATION = "looker_integration"
    STATISTICAL_ANOMALY = "statistical_anomaly"
    MACHINE_LEARNING = "machine_learning"
    PRESENTATION_BUILDING = "presentation_building"
    FUNNEL_ANALYSIS = "funnel_analysis"
    COHORT_ANALYSIS = "cohort_analysis"
    NLP_MODELING = "nlp_modeling"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    SEGMENTATION = "segmentation"
    AB_TESTING = "ab_testing"
    DEEP_ANALYSIS = "deep_analysis"


class DeliverableType(str, Enum):
    ANALYSIS = "analysis"
    DASHBOARD = "dashboard"
    REPORT = "report"
    SLIDES = "slides"
    RULES = "rules"


class Tier(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"


class TaskType(str, Enum):
    REASONING = "reasoning"   # → Claude Sonnet (complex analysis)
    TOOL = "tool"             # → GPT-4o (tool-heavy tasks like SQL generation)
    SIMPLE = "simple"         # → Claude Haiku (cheap, fast: clarification checks)


class Request(BaseModel):
    channel: Channel
    sender_id: str
    sender_name: str
    text: str
    thread_id: Optional[str] = None
    timestamp: str
    client_id: str
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None


class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[str]
    account_mode: str  # "vendor" or "dedicated"
    active_channels: list[Channel]
    database_url: Optional[str] = None
    connector_type: Optional[Literal["postgres", "mysql", "bigquery", "snowflake"]] = None
    connector_config: Optional[dict] = None
    api_key: Optional[str] = None


class ClarificationState(BaseModel):
    original_request: Request
    rounds: int = 0
    questions_asked: list[str] = []
    answers_received: list[str] = []
    is_resolved: bool = False
    assumptions: list[str] = []
    deep_dive_pending: bool = False
    deep_dive_confirmed: bool = False
    deliverable_pending: bool = False
    selected_deliverable: Optional[DeliverableType] = None


class ScheduledJob(BaseModel):
    job_id: str
    client_id: str
    description: str
    request_text: str
    cron_expression: str
    delivery_channel: Channel
    delivery_destination: str
    last_run: Optional[str] = None
    last_result_summary: Optional[str] = None


class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: Optional[dict] = None
    chart_png: Optional[bytes] = None
    deck_pptx: Optional[bytes] = None
    storyline: Optional["Storyline"] = None
    error: Optional[str] = None


class TicketRef(BaseModel):
    key: str
    url: str
    summary: str


class Anomaly(BaseModel):
    metric: str
    value: float
    expected: Optional[float] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    severity: str = "warning"
    description: str
    mode: Literal["hard_rule", "statistical"]


class FindingSlide(BaseModel):
    heading: str
    body: str
    so_what: str
    chart_index: Optional[int] = None


class Solution(BaseModel):
    title: str
    description: str
    pros: list[str] = []
    cons: list[str] = []


class Storyline(BaseModel):
    problem_statement: str
    executive_summary: list[str] = []
    key_findings: list[FindingSlide] = []
    solutions: list[Solution] = []
    recommended_solution: str = ""
    recommendation_rationale: str = ""
    conclusion: str = ""
    next_steps: list[str] = []


class Response(BaseModel):
    request_id: str
    text: str
    charts: list[bytes] = []
    assumptions: list[str] = []
    ticket: Optional[TicketRef] = None
    deck_pptx: Optional[bytes] = None
    anomalies: list[Anomaly] = []
    report_markdown: Optional[str] = None
    report_html: Optional[str] = None


class StepRecord(BaseModel):
    step: int
    thought: str
    tool: str
    tool_input: dict
    observation: str


class AnalystResult(AgentResult):
    steps: list[StepRecord] = []
    findings: list[str] = []
    solutions: list[dict] = []
    recommendation: str = ""


AgentResult.model_rebuild()
