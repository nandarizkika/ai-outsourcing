import io
import json
import logging
from typing import Optional

from pptx import Presentation
from pptx.util import Inches, Pt

from src.core.models import AgentResult, FindingSlide, Solution, Storyline, TaskType

_logger = logging.getLogger(__name__)


class DeckAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(
        self,
        title: str,
        storyline: "Storyline | None" = None,
        sections: "list[dict] | None" = None,
        charts: "list[bytes]" = [],
        template_path: "str | None" = None,
        max_solutions_inline: int = 3,
    ) -> AgentResult:
        import io as _io
        from pptx import Presentation as _Presentation

        try:
            prs = _Presentation(template_path) if template_path else _Presentation()

            if storyline is not None:
                self._add_title_slide(prs, title)
                self._add_executive_summary_slide(prs, storyline)
                self._add_problem_slide(prs, storyline.problem_statement)
                for finding in storyline.key_findings:
                    chart_png = (
                        charts[finding.chart_index]
                        if finding.chart_index is not None and finding.chart_index < len(charts)
                        else None
                    )
                    self._add_finding_slide(prs, finding, chart_png)
                matched = {f.chart_index for f in storyline.key_findings if f.chart_index is not None}
                appendix_charts = [c for i, c in enumerate(charts) if i not in matched]

                if len(storyline.solutions) <= max_solutions_inline:
                    for sol in storyline.solutions:
                        self._add_solution_slide(prs, sol)
                else:
                    self._add_solutions_table_slide(prs, storyline.solutions)

                self._add_recommendation_slide(prs, storyline)
                self._add_conclusion_slide(prs, storyline)

                for i, png in enumerate(appendix_charts):
                    self._add_chart_slide(prs, png, f"Appendix Chart {i + 1}")
            else:
                self._add_title_slide(prs, title)
                for section in (sections or []):
                    self._add_content_slide(prs, section["heading"], section["body"])
                for i, png in enumerate(charts):
                    self._add_chart_slide(prs, png, f"Chart {i + 1}")

            buf = _io.BytesIO()
            prs.save(buf)
            buf.seek(0)
            return AgentResult(
                agent_name="deck_agent",
                success=True,
                deck_pptx=buf.read(),
                storyline=storyline,
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

    def _add_executive_summary_slide(self, prs, storyline) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Executive Summary"
        if len(slide.placeholders) > 1:
            tf = slide.placeholders[1].text_frame
        else:
            tf = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.clear()
        for bullet in storyline.executive_summary:
            p = tf.add_paragraph()
            p.text = bullet
            p.level = 0

    def _add_problem_slide(self, prs, problem_statement: str) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Problem Statement"
        if len(slide.placeholders) > 1:
            tf = slide.placeholders[1].text_frame
        else:
            tf = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.text = problem_statement

    def _add_finding_slide(self, prs, finding, chart_png) -> None:
        import io as _io
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = finding.heading
        if chart_png:
            slide.shapes.add_picture(_io.BytesIO(chart_png), Inches(0.3), Inches(1.2), Inches(5.5), Inches(4.5))
            tb = slide.shapes.add_textbox(Inches(6.0), Inches(1.2), Inches(3.5), Inches(4.5))
            tb.text_frame.text = finding.body + "\n\nSo what: " + finding.so_what
        else:
            if len(slide.placeholders) > 1:
                tf = slide.placeholders[1].text_frame
            else:
                tf = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(5.5), Inches(4)).text_frame
            tf.text = finding.body
            callout = slide.shapes.add_textbox(Inches(6.2), Inches(1.5), Inches(3.3), Inches(1.5))
            callout.text_frame.text = "So what?\n" + finding.so_what

    def _add_solution_slide(self, prs, solution) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = solution.title
        if len(slide.placeholders) > 1:
            tf = slide.placeholders[1].text_frame
        else:
            tf = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.clear()
        p = tf.add_paragraph()
        p.text = solution.description
        p = tf.add_paragraph()
        p.text = "Pros: " + ", ".join(solution.pros)
        p = tf.add_paragraph()
        p.text = "Cons: " + ", ".join(solution.cons)

    def _add_solutions_table_slide(self, prs, solutions) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[5]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Solutions Comparison"
        rows = len(solutions) + 1
        table = slide.shapes.add_table(rows, 3, Inches(0.5), Inches(1.5), Inches(9), Inches(0.5 * rows)).table
        for i, header in enumerate(["Solution", "Pros", "Cons"]):
            table.cell(0, i).text = header
        for r, sol in enumerate(solutions, start=1):
            table.cell(r, 0).text = sol.title
            table.cell(r, 1).text = ", ".join(sol.pros)
            table.cell(r, 2).text = ", ".join(sol.cons)

    def _add_recommendation_slide(self, prs, storyline) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Recommendation"
        tb = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(1.2))
        tb.text_frame.text = storyline.recommended_solution
        if storyline.recommendation_rationale:
            tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(2.8), Inches(9), Inches(2.5))
            tb2.text_frame.text = storyline.recommendation_rationale

    def _add_conclusion_slide(self, prs, storyline) -> None:
        from pptx.util import Inches
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        if slide.shapes.title:
            slide.shapes.title.text = "Conclusion & Next Steps"
        if len(slide.placeholders) > 1:
            tf = slide.placeholders[1].text_frame
        else:
            tf = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(4)).text_frame
        tf.clear()
        if storyline.conclusion:
            p = tf.add_paragraph()
            p.text = storyline.conclusion
        for step in storyline.next_steps:
            p = tf.add_paragraph()
            p.text = "• " + step
            p.level = 1

    def build_storyline(
        self,
        analysis_text: str,
        findings: list[str],
        solutions: list[dict],
        recommendation: str,
    ) -> Storyline:
        if self._llm is None:
            return self._fallback_storyline(analysis_text, findings, solutions, recommendation)
        system = (
            "You are a consulting analyst. Structure the following analysis as a presentation narrative. "
            "Return ONLY valid JSON matching this schema exactly:\n"
            '{"problem_statement": "str", "executive_summary": ["str"], '
            '"key_findings": [{"heading": "str", "body": "str", "so_what": "str", "chart_index": null}], '
            '"solutions": [{"title": "str", "description": "str", "pros": ["str"], "cons": ["str"]}], '
            '"recommended_solution": "str", "recommendation_rationale": "str", '
            '"conclusion": "str", "next_steps": ["str"]}'
        )
        findings_text = "\n".join(findings)
        solutions_text = "\n".join(s.get("title", "") for s in solutions)
        user = (
            f"Analysis:\n{analysis_text}\n\n"
            f"Key findings:\n{findings_text}\n\n"
            f"Solutions considered:\n{solutions_text}\n\n"
            f"Recommendation: {recommendation}"
        )
        raw = self._llm.complete(TaskType.REASONING, system, user)
        try:
            data = json.loads(raw)
            return Storyline(
                problem_statement=data.get("problem_statement", ""),
                executive_summary=data.get("executive_summary", []),
                key_findings=[FindingSlide(**f) for f in data.get("key_findings", [])],
                solutions=[Solution(**s) for s in data.get("solutions", [])],
                recommended_solution=data.get("recommended_solution", recommendation),
                recommendation_rationale=data.get("recommendation_rationale", ""),
                conclusion=data.get("conclusion", ""),
                next_steps=data.get("next_steps", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return self._fallback_storyline(analysis_text, findings, solutions, recommendation)

    def _fallback_storyline(
        self,
        analysis_text: str,
        findings: list[str],
        solutions: list[dict],
        recommendation: str,
    ) -> Storyline:
        paragraphs = [p.strip() for p in analysis_text.split("\n\n") if p.strip()]
        return Storyline(
            problem_statement=paragraphs[0] if paragraphs else analysis_text[:200],
            executive_summary=findings[:5],
            key_findings=[
                FindingSlide(heading=f"Finding {i+1}", body=p, so_what="")
                for i, p in enumerate(paragraphs[:5])
            ],
            solutions=[
                Solution(title=s.get("title", f"Option {i+1}"),
                         description=s.get("description", ""),
                         pros=s.get("pros", []), cons=s.get("cons", []))
                for i, s in enumerate(solutions)
            ],
            recommended_solution=recommendation,
            recommendation_rationale="",
            conclusion="",
            next_steps=[],
        )
