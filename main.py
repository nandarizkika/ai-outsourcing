# main.py
from fastapi import FastAPI, Request, Depends, Header
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
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.segmentation_agent import SegmentationAgent
from src.agents.ab_agent import ABTestingAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.jobs.scheduler import JobScheduler
from src.core.models import ScheduledJob, ClientConfig, ClarificationState
from src.jobs.delivery import deliver
from src.core.client_registry import ClientRegistry
from src.agents.report_agent import ReportAgent
from fastapi import HTTPException
from pydantic import BaseModel as PydanticBaseModel
from typing import Optional as OptionalType
import secrets
from src.core.auth import make_verify_api_key, make_verify_client_api_key
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request as StarletteRequest
from src.core.logging_config import configure_logging
from src.middleware.logging import RequestLoggingMiddleware

settings = Settings()
verify_api_key = make_verify_api_key(settings)
llm = LLMRouter(
    provider=settings.llm_provider,
    gemini_api_key=settings.gemini_api_key,
    openai_api_key=settings.openai_api_key,
    anthropic_api_key=settings.anthropic_api_key,
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
hypothesis_agent = HypothesisAgent(llm=llm)
segmentation_agent = SegmentationAgent(llm=llm)
ab_agent = ABTestingAgent(llm=llm)

registry = ClientRegistry(settings.clients_file)
verify_client_api_key = make_verify_client_api_key(registry)

def _get_client_id(request: StarletteRequest) -> str:
    return request.headers.get("X-Client-ID") or request.client.host

limiter = Limiter(key_func=_get_client_id)


class RegistryAdapter(dict):
    def __getitem__(self, key):
        cfg = registry.get(key)
        if cfg is None:
            raise KeyError(key)
        return cfg

    def __contains__(self, key):
        return registry.get(key) is not None

    def get(self, key, default=None):
        cfg = registry.get(key)
        return cfg if cfg is not None else default


client_configs = RegistryAdapter()

report_agent = ReportAgent(llm=llm)

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
    hypothesis_agent=hypothesis_agent,
    segmentation_agent=segmentation_agent,
    ab_agent=ab_agent,
    registry=registry,
    report_agent=report_agent,
)


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
slack = SlackChannel(
    bot_token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    bot_user_id=settings.slack_bot_user_id,
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

configure_logging(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestLoggingMiddleware)


from src.core.models import Request as AnalystRequest


class AnalyzeBody(PydanticBaseModel):
    request: AnalystRequest
    clarification_state: OptionalType[ClarificationState] = None


@app.post("/clients", status_code=201, dependencies=[Depends(verify_api_key)])
def create_client(config: ClientConfig):
    api_key = secrets.token_hex(32)
    config = config.model_copy(update={"api_key": api_key})
    registry.upsert(config)
    return {"client_id": config.client_id, "api_key": api_key}


@app.get("/clients", dependencies=[Depends(verify_api_key)])
def list_clients():
    return {"clients": [c.model_dump(exclude={"api_key"}) for c in registry.all()]}


@app.get("/clients/{client_id}", dependencies=[Depends(verify_api_key)])
def get_client(client_id: str):
    config = registry.get(client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return config.model_dump(exclude={"api_key"})


@app.delete("/clients/{client_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_client(client_id: str):
    registry.delete(client_id)


@app.post("/clients/{client_id}/rotate-key", dependencies=[Depends(verify_api_key)])
def rotate_client_key(client_id: str):
    config = registry.get(client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    new_key = secrets.token_hex(32)
    registry.upsert(config.model_copy(update={"api_key": new_key}))
    return {"api_key": new_key}


@app.post("/analyze", dependencies=[Depends(verify_client_api_key)])
@limiter.limit("60/minute")
async def analyze(request: StarletteRequest, body: AnalyzeBody, x_client_id: str = Header(default="")):
    if body.request.client_id != x_client_id:
        raise HTTPException(status_code=403, detail="client_id mismatch")
    config = registry.get(body.request.client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    try:
        result = await orchestrator.process(body.request, config, body.clarification_state)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return {"result": str(result)}
    except Exception as e:
        import traceback
        print(f"ERROR in analyze: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/slack/events")
async def slack_events(req: Request):
    return await slack.get_handler().handle(req)


@app.get("/health")
def health():
    from datetime import datetime, timezone
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.now(timezone.utc).isoformat()}


def make_deliver():
    def _deliver(job, response):
        deliver(job, response, slack_channel=slack, email_channel=email_channel)
    return _deliver


@app.post("/jobs", status_code=201, dependencies=[Depends(verify_api_key)])
def create_job(job: ScheduledJob):
    scheduler.add_job(job, on_complete=make_deliver())
    return {"job_id": job.job_id}


@app.get("/jobs", dependencies=[Depends(verify_api_key)])
def list_jobs():
    return {"jobs": [j.model_dump() for j in scheduler.list_jobs()]}


@app.delete("/jobs/{job_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def delete_job(job_id: str):
    scheduler.remove_job(job_id)
