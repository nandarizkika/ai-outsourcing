import json
import uuid

from src.core.llm import LLMRouter
from src.core.models import (
    TaskType, Request, Response, ClientConfig, SkillModule, ClarificationState, AgentResult
)
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.agents.ml_agent import MLAgent
from src.agents.deck_agent import DeckAgent
from src.orchestrator.clarifier import ClarificationChecker


class Orchestrator:
    def __init__(
        self,
        llm: LLMRouter,
        retriever: KnowledgeRetriever,
        clarifier: ClarificationChecker,
        sql_agent: SQLAgent,
        chart_agent: ChartAgent,
        ml_agent: MLAgent | None = None,
        deck_agent: DeckAgent | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._clarifier = clarifier
        self._sql_agent = sql_agent
        self._chart_agent = chart_agent
        self._ml_agent = ml_agent
        self._deck_agent = deck_agent

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

        # ML Agent — runs on SQL data when forecast/prediction requested
        if (
            plan.get("ml")
            and self._ml_agent is not None
            and SkillModule.MACHINE_LEARNING in config.enabled_skills
            and sql_data
        ):
            ml_result = self._ml_agent.run(config.client_id, request.text, sql_data)
            if ml_result.success and ml_result.chart_png:
                charts.append(ml_result.chart_png)

        text = self._generate_response(request, context, sql_data, state.assumptions)

        # Deck Agent — builds PPTX from text + charts
        deck_pptx: bytes | None = None
        if (
            plan.get("deck")
            and self._deck_agent is not None
            and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
        ):
            sections = [{"heading": "Analysis", "body": text}]
            deck_result = self._deck_agent.run(
                title=request.text[:100], sections=sections, charts=charts
            )
            if deck_result.success:
                deck_pptx = deck_result.deck_pptx

        return Response(
            request_id=str(uuid.uuid4()),
            text=text,
            charts=charts,
            assumptions=state.assumptions,
            deck_pptx=deck_pptx,
        )

    def _plan(self, request: Request, context: list[str]) -> dict:
        system = (
            "Determine which capabilities are needed to answer this data request. "
            "Return JSON only: "
            '{"sql": true/false, "chart": true/false, "ml": true/false, "deck": true/false}. '
            "Set ml=true for forecast/predict/projection/trend requests. "
            "Set deck=true for slide/deck/presentation/powerpoint requests."
        )
        user = f"Request: {request.text}\nContext: {chr(10).join(context)}"
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"sql": True, "chart": True, "ml": False, "deck": False}

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
