# tests/test_integration_phase6.py
import io
import json
from unittest.mock import MagicMock
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


def test_orchestrator_routes_to_hypothesis_agent_when_plan_set():
    from src.agents.hypothesis_agent import HypothesisAgent
    import numpy as np

    plan_json = '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": true, "segment": false, "ab_test": false}'
    llm = _make_llm(plan_json=plan_json)
    mock_hyp = MagicMock(spec=HypothesisAgent)
    mock_hyp.run.return_value = MagicMock(success=True, data={"test": "t_test", "p_value": 0.001, "significant": True, "groups": ["A", "B"], "statistic": 5.2, "group_col": "group", "metric_col": "metric", "interpretation": "Significant."})

    rng = np.random.default_rng(42)
    rows = (
        [{"group": "A", "metric": float(v)} for v in rng.normal(10, 1, 20)] +
        [{"group": "B", "metric": float(v)} for v in rng.normal(15, 1, 20)]
    )
    sql_data = {"columns": ["group", "metric"], "rows": rows, "query": "SELECT ..."}

    mock_sql = MagicMock()
    mock_sql.run.return_value = MagicMock(success=True, data=sql_data)

    orch = _make_orchestrator(llm, hypothesis_agent=mock_hyp)
    orch._sql_agent = mock_sql

    config = _make_config([SkillModule.SQL_QUERYING, SkillModule.HYPOTHESIS_TESTING])
    result = orch.process(_make_request("Is there a significant difference between groups A and B?"), config)
    mock_hyp.run.assert_called_once()


def test_orchestrator_routes_to_segmentation_agent_when_plan_set():
    from src.agents.segmentation_agent import SegmentationAgent
    import numpy as np

    plan_json = '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": true, "ab_test": false}'
    llm = _make_llm(plan_json=plan_json)
    mock_seg = MagicMock(spec=SegmentationAgent)
    mock_seg.run.return_value = MagicMock(success=True, chart_png=b"PNG", data={"n_segments": 2, "centroids": {}, "silhouette_score": 0.7, "rows": [], "columns": [], "interpretation": ""})

    rng = np.random.default_rng(42)
    rows = [{"revenue": float(r), "sessions": float(s)}
            for r, s in zip(rng.normal(100, 10, 20), rng.normal(50, 5, 20))]
    sql_data = {"columns": ["revenue", "sessions"], "rows": rows, "query": "SELECT ..."}

    mock_sql = MagicMock()
    mock_sql.run.return_value = MagicMock(success=True, data=sql_data)

    orch = _make_orchestrator(llm, segmentation_agent=mock_seg)
    orch._sql_agent = mock_sql

    config = _make_config([SkillModule.SQL_QUERYING, SkillModule.SEGMENTATION])
    result = orch.process(_make_request("Segment our customers by revenue and sessions"), config)
    mock_seg.run.assert_called_once()


def test_orchestrator_routes_to_ab_agent_when_plan_set():
    from src.agents.ab_agent import ABTestingAgent

    plan_json = '{"sql": true, "chart": false, "ml": false, "deck": false, "funnel": false, "cohort": false, "hypothesis": false, "segment": false, "ab_test": true}'
    llm = _make_llm(plan_json=plan_json)
    mock_ab = MagicMock(spec=ABTestingAgent)
    mock_ab.run.return_value = MagicMock(success=True, data={"test": "proportions_ztest", "uplift": 0.05, "significant": True, "control_mean": 0.1, "variant_mean": 0.15, "statistic": 3.1, "p_value": 0.002, "variant_col": "variant", "metric_col": "converted", "recommendation": "Ship."})

    rows = (
        [{"variant": "control", "converted": 1} for _ in range(10)] +
        [{"variant": "control", "converted": 0} for _ in range(90)] +
        [{"variant": "treatment", "converted": 1} for _ in range(15)] +
        [{"variant": "treatment", "converted": 0} for _ in range(85)]
    )
    sql_data = {"columns": ["variant", "converted"], "rows": rows, "query": "SELECT ..."}

    mock_sql = MagicMock()
    mock_sql.run.return_value = MagicMock(success=True, data=sql_data)

    orch = _make_orchestrator(llm, ab_agent=mock_ab)
    orch._sql_agent = mock_sql

    config = _make_config([SkillModule.SQL_QUERYING, SkillModule.AB_TESTING])
    result = orch.process(_make_request("Evaluate our A/B test results"), config)
    mock_ab.run.assert_called_once()
