# tests/test_integration_phase6.py
import io
import json
from unittest.mock import MagicMock, patch
import pytest
from pptx import Presentation

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.clarifier import ClarificationChecker
from src.agents.deck_agent import DeckAgent
from src.core.models import ClientConfig, Tier, SkillModule, Channel, Request


def _make_llm(plan_json=None, response_text="Analysis complete."):
    mock = MagicMock()
    def complete(task_type, system, user):
        if "which capabilities" in system.lower():
            return plan_json or '{"sql": false, "chart": false, "ml": false, "deck": true, "funnel": false, "cohort": false}'
        if "consulting analyst" in system.lower() or "structure the following" in system.lower():
            return json.dumps({
                "problem_statement": "Revenue dropped.",
                "executive_summary": ["Revenue down 10%"],
                "key_findings": [{"heading": "Drop", "body": "Revenue fell.", "so_what": "Act now.", "chart_index": None}],
                "solutions": [],
                "recommended_solution": "",
                "recommendation_rationale": "",
                "conclusion": "Act now.",
                "next_steps": ["Fix it"],
            })
        if "root-cause" in system.lower() or "deep investigation" in system.lower():
            return "false"
        if "enough information" in system.lower() or "clarif" in system.lower():
            return json.dumps({"resolved": True, "questions": [], "assumptions": []})
        return response_text
    mock.complete.side_effect = complete
    return mock


def _make_config(skills):
    return ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=skills, account_mode="vendor",
        active_channels=[Channel.SLACK],
    )


def _make_request(text, client_id="c1"):
    return Request(
        channel=Channel.SLACK, sender_id="u1", sender_name="User",
        text=text, timestamp="2026-05-16T00:00:00Z", client_id=client_id,
    )


def _make_orchestrator(llm, **agents):
    from src.knowledge.retriever import KnowledgeRetriever
    from src.agents.chart_agent import ChartAgent
    from src.agents.sql_agent import SQLAgent
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    clarifier = ClarificationChecker(llm=llm)
    return Orchestrator(
        llm=llm,
        retriever=mock_retriever,
        clarifier=clarifier,
        sql_agent=MagicMock(spec=SQLAgent),
        chart_agent=MagicMock(spec=ChartAgent),
        **agents,
    )


def test_orchestrator_deck_uses_storyline_path():
    llm = _make_llm()
    deck_agent = DeckAgent(llm=llm)
    orch = _make_orchestrator(llm, deck_agent=deck_agent)
    config = _make_config([SkillModule.PRESENTATION_BUILDING])
    result = orch.process(_make_request("Build me a presentation of the analysis"), config)
    assert hasattr(result, "deck_pptx")
    assert result.deck_pptx is not None
    prs = Presentation(io.BytesIO(result.deck_pptx))
    # Storyline path produces title + exec summary + problem + findings + recommendation + conclusion
    assert len(prs.slides) >= 4
