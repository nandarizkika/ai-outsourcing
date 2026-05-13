from enum import Enum
from typing import Optional
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


class ClientConfig(BaseModel):
    client_id: str
    name: str
    tier: Tier
    enabled_skills: list[SkillModule]
    account_mode: str  # "vendor" or "dedicated"
    active_channels: list[Channel]


class ClarificationState(BaseModel):
    original_request: Request
    rounds: int = 0
    questions_asked: list[str] = []
    answers_received: list[str] = []
    is_resolved: bool = False
    assumptions: list[str] = []


class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: Optional[dict] = None
    chart_png: Optional[bytes] = None
    error: Optional[str] = None


class TicketRef(BaseModel):
    key: str
    url: str
    summary: str


class Response(BaseModel):
    request_id: str
    text: str
    charts: list[bytes] = []
    assumptions: list[str] = []
    ticket: Optional[TicketRef] = None
