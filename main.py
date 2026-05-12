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

# Client configs loaded here — in production, load from a config file or DB
client_configs = {}  # workspace_team_id -> ClientConfig

orchestrator = Orchestrator(
    llm=llm,
    retriever=retriever,
    clarifier=clarifier,
    sql_agent=None,   # injected per-client at runtime
    chart_agent=chart_agent,
)

# Slack bot user ID — fetch from Slack API at startup
SLACK_BOT_USER_ID = "REPLACE_WITH_BOT_USER_ID"

slack = SlackChannel(
    bot_token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
    bot_user_id=SLACK_BOT_USER_ID,
    orchestrator=orchestrator,
    client_configs=client_configs,
)

app = FastAPI()


@app.post("/slack/events")
async def slack_events(req: Request):
    return await slack.get_handler().handle(req)


@app.get("/health")
def health():
    return {"status": "ok"}
