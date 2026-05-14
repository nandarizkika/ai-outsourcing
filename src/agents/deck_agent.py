import io
import logging
from typing import Optional

from pptx import Presentation
from pptx.util import Inches, Pt

from src.core.models import AgentResult

_logger = logging.getLogger(__name__)


class DeckAgent:
    def run(
        self,
        title: str,
        sections: list[dict],
        charts: list[bytes],
        template_path: Optional[str] = None,
    ) -> AgentResult:
        try:
            prs = Presentation(template_path) if template_path else Presentation()
            self._add_title_slide(prs, title)
            for section in sections:
                self._add_content_slide(prs, section["heading"], section["body"])
            for i, png in enumerate(charts):
                self._add_chart_slide(prs, png, f"Chart {i + 1}")
            buf = io.BytesIO()
            prs.save(buf)
            buf.seek(0)
            return AgentResult(
                agent_name="deck_agent",
                success=True,
                deck_pptx=buf.read(),
            )
        except Exception as exc:
            _logger.error("DeckAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="deck_agent", success=False, error=str(exc))

    def _add_title_slide(self, prs: Presentation, title: str) -> None:
        layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = title
        if len(slide.placeholders) > 1:
            slide.placeholders[1].text = "AI Data Analyst"

    def _add_content_slide(self, prs: Presentation, heading: str, body: str) -> None:
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = heading
        if len(slide.placeholders) > 1:
            slide.placeholders[1].text = body

    def _add_chart_slide(self, prs: Presentation, png_bytes: bytes, caption: str) -> None:
        layout = prs.slide_layouts[6]  # blank
        slide = prs.slides.add_slide(layout)
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
        txBox.text_frame.text = caption
        slide.shapes.add_picture(
            io.BytesIO(png_bytes),
            Inches(0.5),
            Inches(1.0),
            Inches(9),
            Inches(5.5),
        )
