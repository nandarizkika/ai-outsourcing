# main.py
from fastapi import FastAPI, Request
from src.core.config import Settings
from src.core.llm import LLMRouter
from src.knowledge.vector_store import VectorStore
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.channels.slack import SlackChannel
from src.channels.email_channel import EmailChannel
from src.ticketing.jira_client import JiraClient
from src.ticketing.ticketing_service import TicketingService
from contextlib import asynccontextmanager
from src.agents.ml_agent import MLAgent
from src.agents.deck_agent import DeckAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.spreadsheet_agent import SpreadsheetAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.jobs.scheduler import JobScheduler
from src.core.models import ScheduledJob

settings = Settings()
llm = LLMRouter(
    anthropic_api_key=settings.anthropic_api_key,
    openai_api_key=settings.openai_api_key,
)
store = VectorStore(
    persist_dir=settings.chromadb_persist_dir,
    openai_api_key=settings.openai_api_key,
)
retriever = KnowledgeRetriever(store=store)
clarifier = ClarificationChecker(llm=llm)
chart_agent = ChartAgent()
ml_agent = MLAgent(llm=llm)
deck_agent = DeckAgent()
anomaly_agent = AnomalyAgent(retriever=retriever)
memory_logger = InteractionMemoryLogger(store=store)
analyst_agent = AnalystAgent(llm=llm, sql_agent=None, retriever=retriever, ml_agent=ml_agent)
spreadsheet_agent = SpreadsheetAgent(llm=llm)

# Client configs loaded here — in production, load from a config file or DB
client_configs = {}  # workspace_team_id -> ClientConfig

orchestrator = Orchestrator(
    llm=llm,
    retriever=retriever,
    clarifier=clarifier,
    sql_agent=None,
    chart_agent=chart_agent,
    ml_agent=ml_agent,
    deck_agent=deck_agent,
    anomaly_agent=anomaly_agent,
    memory_logger=memory_logger,
    analyst_agent=analyst_agent,
    spreadsheet_agent=spreadsheet_agent,
)


def _noop_delivery(response):
    pass


scheduler = JobScheduler(
    orchestrator=orchestrator,
    client_configs=client_configs,
)
# Phase 2: Ticketing (Jira) — only wired when jira_url is configured
ticketing_service = None
if settings.jira_url:
    jira_client = JiraClient(
        jira_url=settings.jira_url,
        email=settings.jira_email,
        api_token=settings.jira_api_token,
        project_key=settings.jira_project_key,
    )
    ticketing_service = TicketingService(jira_client=jira_client)

# Slack bot user ID — fetch from Slack API at startup
SLACK_BOT_USER_ID = "REPLACE_WITH_BOT_USER_ID"

slack = SlackChannel(
    bot_token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    bot_user_id=SLACK_BOT_USER_ID,
    orchestrator=orchestrator,
    client_configs=client_configs,
    ticketing_service=ticketing_service,
)

# Phase 2: Email channel — only wired when email_imap_host is configured
email_channel = None
if settings.email_imap_host:
    email_channel = EmailChannel(
        imap_host=settings.email_imap_host,
        imap_user=settings.email_imap_user,
        imap_password=settings.email_imap_password,
        smtp_host=settings.email_smtp_host,
        smtp_port=settings.email_smtp_port,
        orchestrator=orchestrator,
        client_configs=client_configs,
        ticketing_service=ticketing_service,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/slack/events")
async def slack_events(req: Request):
    return await slack.get_handler().handle(req)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/jobs", status_code=201)
def create_job(job: ScheduledJob):
    scheduler.add_job(job, on_complete=_noop_delivery)
    return {"job_id": job.job_id}


@app.get("/jobs")
def list_jobs():
    return {"jobs": [j.model_dump() for j in scheduler.list_jobs()]}


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    scheduler.remove_job(job_id)
