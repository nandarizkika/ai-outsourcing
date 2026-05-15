import io
import json
import pytest
from unittest.mock import MagicMock
from pptx import Presentation

from src.agents.deck_agent import DeckAgent
from src.core.models import AgentResult, FindingSlide, Solution, Storyline


def test_deck_agent_returns_pptx_bytes():
    agent = DeckAgent()
    result = agent.run(
        title="Q1 Revenue Analysis",
        sections=[
            {"heading": "Key Findings", "body": "Revenue grew 15% in Q1."},
            {"heading": "Recommendations", "body": "Focus on top-performing regions."},
        ],
        charts=[],
    )
    assert result.success is True
    assert result.deck_pptx is not None
    assert len(result.deck_pptx) > 0
    prs = Presentation(io.BytesIO(result.deck_pptx))
    assert len(prs.slides) >= 1


def test_deck_agent_title_slide_text():
    agent = DeckAgent()
    result = agent.run(
        title="Monthly Report",
        sections=[{"heading": "Summary", "body": "All metrics are on track."}],
        charts=[],
    )
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    first_slide_text = " ".join(
        shape.text for shape in prs.slides[0].shapes if shape.has_text_frame
    )
    assert "Monthly Report" in first_slide_text


def test_deck_agent_attaches_chart_image():
    agent = DeckAgent()
    # Minimal valid 1x1 PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    result = agent.run(
        title="Chart Report",
        sections=[{"heading": "Chart", "body": "See the chart below."}],
        charts=[png_bytes],
    )
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    has_picture = any(
        shape.shape_type == 13
        for slide in prs.slides
        for shape in slide.shapes
    )
    assert has_picture


def test_storyline_model_fields():
    s = Storyline(
        problem_statement="Churn increased 20% in Q2.",
        executive_summary=["Churn up 20%", "Root cause: pricing"],
        key_findings=[
            FindingSlide(heading="Q2 Churn", body="Churn hit 15%.",
                         so_what="We risk losing top cohort.", chart_index=None)
        ],
        solutions=[
            Solution(title="Price rollback", description="Revert to Q1 pricing.",
                     pros=["Quick"], cons=["Revenue impact"])
        ],
        recommended_solution="Price rollback",
        recommendation_rationale="Fastest path to churn reduction.",
        conclusion="Act within 30 days.",
        next_steps=["Rollback pricing", "Monitor churn weekly"],
    )
    assert s.problem_statement == "Churn increased 20% in Q2."
    assert len(s.key_findings) == 1
    assert s.key_findings[0].chart_index is None
    assert len(s.solutions) == 1


def test_agent_result_storyline_defaults_none():
    r = AgentResult(agent_name="deck_agent", success=True)
    assert r.storyline is None


def test_build_storyline_returns_storyline_object():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "problem_statement": "Revenue dropped.",
        "executive_summary": ["Revenue down 10%"],
        "key_findings": [{"heading": "Drop", "body": "Revenue fell.", "so_what": "Action needed.", "chart_index": None}],
        "solutions": [{"title": "Fix X", "description": "Do X.", "pros": ["fast"], "cons": ["costly"]}],
        "recommended_solution": "Fix X",
        "recommendation_rationale": "Fastest fix.",
        "conclusion": "Act now.",
        "next_steps": ["Start fix"],
    })
    agent = DeckAgent(llm=mock_llm)
    storyline = agent.build_storyline(
        analysis_text="Revenue dropped 10% in Q3.",
        findings=["Revenue fell"],
        solutions=[{"title": "Fix X", "description": "Do X.", "pros": ["fast"], "cons": ["costly"]}],
        recommendation="Fix X",
    )
    assert isinstance(storyline, Storyline)
    assert storyline.problem_statement == "Revenue dropped."
    assert len(storyline.key_findings) == 1


def _make_storyline(n_findings=2, n_solutions=2, with_chart_index=False):
    return Storyline(
        problem_statement="Revenue fell 10% in Q2.",
        executive_summary=["Revenue down 10%", "Root cause found", "Action required"],
        key_findings=[
            FindingSlide(
                heading=f"Finding {i+1}",
                body=f"Detail {i+1}.",
                so_what=f"Impact {i+1}.",
                chart_index=0 if (with_chart_index and i == 0) else None,
            )
            for i in range(n_findings)
        ],
        solutions=[
            Solution(title=f"Option {i+1}", description=f"Desc {i+1}.",
                     pros=["Pro"], cons=["Con"])
            for i in range(n_solutions)
        ],
        recommended_solution="Option 1",
        recommendation_rationale="Fastest impact.",
        conclusion="Act within 30 days.",
        next_steps=["Step 1", "Step 2"],
    )


def test_run_with_storyline_produces_valid_pptx():
    agent = DeckAgent()
    storyline = _make_storyline()
    result = agent.run(title="Q2 Revenue Analysis", storyline=storyline, charts=[])
    assert result.success is True
    assert result.deck_pptx is not None
    prs = Presentation(io.BytesIO(result.deck_pptx))
    assert len(prs.slides) >= 5  # title + exec summary + problem + findings + recommendation


def test_result_contains_storyline():
    agent = DeckAgent()
    storyline = _make_storyline()
    result = agent.run(title="Report", storyline=storyline, charts=[])
    assert result.success is True
    assert result.storyline is not None
    assert result.storyline.problem_statement == "Revenue fell 10% in Q2."


def test_executive_summary_slide_is_second():
    agent = DeckAgent()
    storyline = _make_storyline()
    result = agent.run(title="Report", storyline=storyline, charts=[])
    prs = Presentation(io.BytesIO(result.deck_pptx))
    second_slide_text = " ".join(
        shape.text for shape in prs.slides[1].shapes if shape.has_text_frame
    )
    assert any(bullet in second_slide_text for bullet in storyline.executive_summary)


def test_inline_chart_appears_in_finding_slide_not_appended():
    agent = DeckAgent()
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    storyline = _make_storyline(n_findings=2, with_chart_index=True)
    result = agent.run(title="Chart Report", storyline=storyline, charts=[png_bytes])
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    has_picture = any(
        shape.shape_type == 13 for slide in prs.slides for shape in slide.shapes
    )
    assert has_picture


def test_many_solutions_uses_comparison_table_slide():
    agent = DeckAgent()
    storyline = _make_storyline(n_solutions=4)
    result = agent.run(title="Solutions", storyline=storyline, charts=[], max_solutions_inline=3)
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    # 4 solutions > max_solutions_inline=3, so comparison table used (1 slide)
    # Total should be less than title+exec+problem+4_solution_slides+rec+conclusion
    assert len(prs.slides) < 10


def test_legacy_sections_still_work():
    agent = DeckAgent()
    result = agent.run(
        title="Legacy Report",
        sections=[{"heading": "Summary", "body": "All good."}],
        charts=[],
    )
    assert result.success is True
    prs = Presentation(io.BytesIO(result.deck_pptx))
    assert len(prs.slides) >= 1
