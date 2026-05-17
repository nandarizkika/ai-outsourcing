import logging
import markdown as md_lib

from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class ReportAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            analysis_text = data.get("analysis_text", "")
            sql_rows = data.get("sql_rows", [])
            anomalies = data.get("anomalies", [])

            rows_text = ""
            if sql_rows:
                headers = list(sql_rows[0].keys())
                rows_text = "| " + " | ".join(headers) + " |\n"
                rows_text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in sql_rows[:10]:
                    rows_text += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"

            anomaly_text = ""
            if anomalies:
                anomaly_text = "\n".join(
                    f"- **{a.get('metric', 'metric')}**: {a.get('description', '')}"
                    for a in anomalies
                )

            system = (
                "You are a business analyst. Write a concise report in Markdown with these sections:\n"
                "## Executive Summary\n(3-5 bullet points)\n"
                "## Key Findings\n(use the data table if provided)\n"
                "## Anomalies\n(only if anomalies listed; omit entirely if none)\n"
                "## Recommendation\n(1-2 sentence action)\n"
                "Return ONLY the Markdown — no preamble."
            )
            user = f"Request: {request}\n\nAnalysis:\n{analysis_text}\n\n"
            if rows_text:
                user += f"Data:\n{rows_text}\n\n"
            if anomaly_text:
                user += f"Anomalies:\n{anomaly_text}\n\n"

            report_md = self._llm.complete(TaskType.REASONING, system, user)
            report_html = md_lib.markdown(report_md, extensions=["tables"])

            return AgentResult(
                agent_name="report_agent",
                success=True,
                data={"markdown": report_md, "html": report_html},
            )
        except Exception as exc:
            _logger.error("ReportAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="report_agent", success=False, error=str(exc))
