from src.core.models import Response, Request, Anomaly
from typing import Optional


class SlackFormatter:
    @staticmethod
    def format_analysis(response: Response, request: Request, sql_data: Optional[dict] = None) -> str:
        lines = []

        title = request.text[:80] if request.text else "Analysis"
        lines.append(f"*📊 ANALYSIS: {title}*\n")

        lines.append("*Key Metrics*")
        if sql_data and sql_data.get("rows"):
            rows = sql_data.get("rows", [])
            lines.append(f"• Total records: {len(rows):,}")

            for row in rows[:3]:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key not in ["user_id", "id"]:
                        lines.append(f"• {key}: {value}")
                break

        lines.append("")
        lines.append("*Findings*")
        lines.append(response.text)

        if response.assumptions:
            lines.append("")
            lines.append("*Assumptions Made*")
            for i, assumption in enumerate(response.assumptions, 1):
                lines.append(f"{i}. {assumption}")

        if sql_data:
            lines.append("")
            lines.append("*Data Source*")
            rows_count = len(sql_data.get("rows", []))
            lines.append(f"Rows analyzed: {rows_count:,}")

        return "\n".join(lines)

    @staticmethod
    def format_dashboard(response: Response, request: Request, sql_data: Optional[dict] = None) -> str:
        lines = []

        title = request.text[:80] if request.text else "Dashboard"
        lines.append(f"*📈 DASHBOARD: {title}*\n")

        lines.append("*Key Metrics*")
        if sql_data and sql_data.get("rows"):
            rows = sql_data.get("rows", [])
            lines.append(f"• Total records: {len(rows):,}")

            for row in rows[:5]:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key not in ["user_id", "id"]:
                        lines.append(f"• {key}: {value}")
                break

        lines.append("")
        lines.append("*Visualization Summary*")
        lines.append(response.text)

        if sql_data:
            lines.append("")
            lines.append("*Data Overview*")
            rows_count = len(sql_data.get("rows", []))
            lines.append(f"Records: {rows_count:,}")

        return "\n".join(lines)

    @staticmethod
    def format_report(response: Response, request: Request) -> str:
        lines = []

        title = request.text[:80] if request.text else "Report"
        lines.append(f"*📋 REPORT: {title}*\n")

        if response.report_markdown:
            lines.append(response.report_markdown)
        else:
            lines.append(response.text)

        if response.assumptions:
            lines.append("")
            lines.append("*Assumptions Made*")
            for i, assumption in enumerate(response.assumptions, 1):
                lines.append(f"{i}. {assumption}")

        return "\n".join(lines)

    @staticmethod
    def format_rules(response: Response, request: Request) -> str:
        lines = []

        title = request.text[:80] if request.text else "Business Rules"
        lines.append(f"*📌 ALERTS & RULES: {title}*\n")

        if response.anomalies:
            lines.append("*Detected Anomalies*")
            for anomaly in response.anomalies[:10]:
                severity_emoji = "🔴" if anomaly.severity == "critical" else "🟡" if anomaly.severity == "warning" else "🔵"
                lines.append(f"{severity_emoji} {anomaly.metric}: {anomaly.value} (expected: {anomaly.expected})")

        lines.append("")
        lines.append("*Analysis*")
        lines.append(response.text)

        if response.assumptions:
            lines.append("")
            lines.append("*Assumptions*")
            for assumption in response.assumptions[:5]:
                lines.append(f"• {assumption}")

        return "\n".join(lines)

    @staticmethod
    def format_slides(response: Response, request: Request) -> str:
        lines = []

        title = request.text[:80] if request.text else "Presentation"
        lines.append(f"*🎨 PRESENTATION: {title}*\n")

        lines.append("Presentation deck has been generated and is ready for download.")
        lines.append("")
        lines.append("*Summary*")
        lines.append(response.text)

        return "\n".join(lines)
