"""
Full flow: Slack mention → Orchestrator → SQL Agent → Chart Agent → Slack reply.
All external calls (LLM, DB, Slack) are mocked.
"""
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.core.models import Channel, ClientConfig, Tier, Response, ClarificationState
from src.core.llm import LLMRouter
from src.knowledge.vector_store import VectorStore
from src.knowledge.retriever import KnowledgeRetriever
from src.connectors.sql import SQLConnector
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker
from src.orchestrator.orchestrator import Orchestrator
from src.skill_modules import SkillModuleRegistry


@pytest.fixture
def full_system(tmp_path, mocker):
    # Mock LLM — 4 calls in order:
    # 1. ClarificationChecker.check  → is_clear
    # 2. Orchestrator._plan          → plan JSON
    # 3. SQLAgent.run                → raw SQL
    # 4. Orchestrator._generate_response → response text
    llm = MagicMock(spec=LLMRouter)
    llm.complete.side_effect = [
        '{"is_clear": true, "questions": []}',           # clarifier check
        '{"sql": true, "chart": true}',                   # orchestrator plan
        "SELECT region, amount FROM sales",               # SQL agent generates SQL
        "Total sales by region for last month:\n- Jakarta: IDR 1,000,000\n- Surabaya: IDR 2,000,000",  # response text
    ]

    # Real VectorStore (tmp dir), mock the embedding wrapper so no OpenAI call is made.
    # The _OpenAIEmbeddingWrapper.__call__ is the actual call path used by ChromaDB.
    mocker.patch(
        "src.knowledge.vector_store._OpenAIEmbeddingWrapper.__call__",
        return_value=[[0.1] * 1536],
    )
    store = VectorStore(persist_dir=str(tmp_path), openai_api_key="fake")
    # Patch store.search and store.add to avoid any live embedding calls.
    # SQLAgent.run calls store.add after a successful query to cache the SQL template.
    mocker.patch.object(store, "search", return_value=[])
    mocker.patch.object(store, "add", return_value=None)
    retriever = KnowledgeRetriever(store=store)

    # Mock SQL connector
    connector = MagicMock(spec=SQLConnector)
    connector.get_schema.return_value = {
        "sales": [
            {"name": "region", "type": "TEXT"},
            {"name": "amount", "type": "REAL"},
        ]
    }
    connector.execute.return_value = pd.DataFrame([
        {"region": "Jakarta", "amount": 1_000_000},
        {"region": "Surabaya", "amount": 2_000_000},
    ])

    sql_agent = SQLAgent(llm=llm, connector=connector, store=store)
    chart_agent = ChartAgent()
    clarifier = ClarificationChecker(llm=llm)
    orchestrator = Orchestrator(
        llm=llm,
        retriever=retriever,
        clarifier=clarifier,
        sql_agent=sql_agent,
        chart_agent=chart_agent,
    )

    config = ClientConfig(
        client_id="client-1",
        name="Test Corp",
        tier=Tier.ADVANCED,
        enabled_skills=SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED),
        account_mode="vendor",
        active_channels=[Channel.SLACK],
    )

    return orchestrator, config


def make_request(text: str = "show total sales by region for last month"):
    from src.core.models import Request
    return Request(
        channel=Channel.SLACK,
        sender_id="U123",
        sender_name="Budi",
        text=text,
        thread_id="thread-001",
        timestamp=datetime.utcnow().isoformat(),
        client_id="client-1",
    )


def test_clear_request_returns_response_with_chart(full_system):
    orchestrator, config = full_system
    result = orchestrator.process(make_request(), config)

    assert isinstance(result, Response)
    assert "Jakarta" in result.text or "sales" in result.text.lower()
    assert len(result.charts) == 1
    assert result.charts[0][:4] == b"\x89PNG"


def test_ambiguous_request_returns_clarification(full_system, mocker):
    orchestrator, config = full_system
    orchestrator._clarifier._llm.complete.side_effect = [
        '{"is_clear": false, "questions": ["Which time period?", "All regions or specific?"]}',
    ]
    result = orchestrator.process(make_request("show me the data"), config)

    assert isinstance(result, ClarificationState)
    assert result.is_resolved is False
    assert len(result.questions_asked) == 2
