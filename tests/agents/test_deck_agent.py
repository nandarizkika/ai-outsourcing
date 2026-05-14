import io
import pytest
from pptx import Presentation

from src.agents.deck_agent import DeckAgent
from src.core.models import AgentResult


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
