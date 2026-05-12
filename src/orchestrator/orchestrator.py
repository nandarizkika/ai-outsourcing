import json
import uuid

from src.core.llm import LLMRouter
from src.core.models import (
    TaskType, Request, Response, ClientConfig, SkillModule, ClarificationState, AgentResult
)
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.orchestrator.clarifier import ClarificationChecker


class Orchestrator:
    def __init__(
        self,
        llm: LLMRouter,
        retriever: KnowledgeRetriever,
        clarifier: ClarificationChecker,
        sql_agent: SQLAgent,
        chart_agent: ChartAgent,
    ):
        self._llm = llm
        self._retriever = retriever
        self._clarifier = clarifier
        self._sql_agent = sql_agent
        self._chart_agent = chart_agent

    def process(
        self,
        request: Request,
        config: ClientConfig,
        clarification_state: ClarificationState | None = None,
    ) -> Response | ClarificationState:
        context = self._retriever.search(config.client_id, request.text)
        state = self._clarifier.check(request, context, clarification_state)

        if not state.is_resolved:
            return state

        plan = self._plan(request, context)
        charts: list[bytes] = []
        sql_data: dict | None = None

        if plan.get("sql") and SkillModule.SQL_QUERYING in config.enabled_skills:
            sql_result = self._sql_agent.run(config.client_id, request.text, context)
            if sql_result.success:
                sql_data = sql_result.data

                if plan.get("chart") and SkillModule.DATA_VISUALIZATION in config.enabled_skills and sql_data:
                    chart_result = self._chart_agent.run(sql_data)
                    if chart_result.success and chart_result.chart_png:
                        charts.append(chart_result.chart_png)

        text = self._generate_response(request, context, sql_data, state.assumptions)

        return Response(
            request_id=str(uuid.uuid4()),
            text=text,
            charts=charts,
            assumptions=state.assumptions,
        )

    def _plan(self, request: Request, context: list[str]) -> dict:
        system = (
            "Determine which capabilities are needed to answer this data request. "
            'Return JSON only: {"sql": true/false, "chart": true/false}'
        )
        user = f"Request: {request.text}\nContext: {chr(10).join(context)}"
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"sql": True, "chart": True}

    def _generate_response(
        self,
        request: Request,
        context: list[str],
        sql_data: dict | None,
        assumptions: list[str],
    ) -> str:
        data_summary = ""
        if sql_data and sql_data.get("rows"):
            rows = sql_data["rows"][:10]
            data_summary = f"\nQuery results (first {len(rows)} rows):\n{json.dumps(rows, default=str)}"

        assumptions_str = ""
        if assumptions:
            assumptions_str = f"\nAssumptions made: {', '.join(assumptions)}\n"

        system = (
            "You are a professional AI data analyst. Write a clear, insightful analysis. "
            "If assumptions were made, state them at the start of your response."
        )
        user = (
            f"Request: {request.text}\n"
            f"{assumptions_str}"
            f"Business context:\n{chr(10).join(context)}"
            f"{data_summary}\n\n"
            "Write the analysis:"
        )
        return self._llm.complete(TaskType.REASONING, system, user)
