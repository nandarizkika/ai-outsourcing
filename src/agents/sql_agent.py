# src/agents/sql_agent.py
import json
import uuid

from src.core.llm import LLMRouter
from src.core.models import TaskType, AgentResult
from src.connectors.sql import SQLConnector
from src.knowledge.vector_store import VectorStore


class SQLAgent:
    def __init__(self, llm: LLMRouter, connector: SQLConnector, store: VectorStore):
        self._llm = llm
        self._connector = connector
        self._store = store

    def run(
        self,
        client_id: str,
        request: str,
        context: list[str],
        analysis_mode: str | None = None,
    ) -> AgentResult:
        schema = self._connector.get_schema()
        hint = ""
        if analysis_mode == "funnel":
            hint = (
                "\nFunnel analysis: write a query that computes conversion rates "
                "and drop-off counts for each ordered stage."
            )
        elif analysis_mode == "cohort":
            hint = (
                "\nCohort analysis: write a query that groups users by acquisition "
                "period and computes retention rates by period offset."
            )
        system = (
            "You are a SQL expert. Given a user request, database schema, and business context, "
            "generate a single valid read-only SQL SELECT query. "
            "Return ONLY the SQL query — no explanation, no markdown, no backticks."
            + hint
        )
        user = (
            f"Request: {request}\n\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Business context:\n{chr(10).join(context)}\n\n"
            "Write the SQL SELECT query:"
        )

        sql = self._llm.complete(TaskType.TOOL, system, user).strip()

        try:
            df = self._connector.execute(sql)
            self._store.add(
                client_id,
                [f"Successful SQL for: {request}\nQuery: {sql}"],
                [str(uuid.uuid4())],
                [{"type": "sql_template", "client_id": client_id}],
            )
            return AgentResult(
                agent_name="sql_agent",
                success=True,
                data={
                    "query": sql,
                    "rows": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                },
            )
        except Exception as e:
            return AgentResult(agent_name="sql_agent", success=False, error=str(e))
