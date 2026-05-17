import json
import uuid

from src.core.llm import LLMRouter
from src.core.models import (
    Anomaly, TaskType, Request, Response, ClientConfig, SkillModule,
    ClarificationState, AgentResult, AnalystResult,
)
from src.knowledge.retriever import KnowledgeRetriever
from src.agents.sql_agent import SQLAgent
from src.agents.chart_agent import ChartAgent
from src.agents.ml_agent import MLAgent
from src.agents.deck_agent import DeckAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.spreadsheet_agent import SpreadsheetAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.segmentation_agent import SegmentationAgent
from src.agents.ab_agent import ABTestingAgent
from src.knowledge.interaction_memory import InteractionMemoryLogger
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
        anomaly_agent: AnomalyAgent | None = None,
        memory_logger: InteractionMemoryLogger | None = None,
        analyst_agent: AnalystAgent | None = None,
        spreadsheet_agent: SpreadsheetAgent | None = None,
        hypothesis_agent: HypothesisAgent | None = None,
        segmentation_agent: SegmentationAgent | None = None,
        ab_agent: ABTestingAgent | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._clarifier = clarifier
        self._sql_agent = sql_agent
        self._chart_agent = chart_agent
        self._ml_agent = ml_agent
        self._deck_agent = deck_agent
        self._anomaly_agent = anomaly_agent
        self._memory_logger = memory_logger
        self._analyst_agent = analyst_agent
        self._spreadsheet_agent = spreadsheet_agent
        self._hypothesis_agent = hypothesis_agent
        self._segmentation_agent = segmentation_agent
        self._ab_agent = ab_agent

    def _detect_deep_intent(self, request: Request) -> bool:
        system = (
            "Does this request ask for root-cause analysis, deep investigation, "
            "or a complex multi-step analytical inquiry? "
            "Return only 'true' or 'false'."
        )
        raw = self._llm.complete(TaskType.SIMPLE, system, f"Request: {request.text}")
        return raw.strip().lower().startswith("true")

    def process(
        self,
        request: Request,
        config: ClientConfig,
        clarification_state: ClarificationState | None = None,
    ) -> "Response | ClarificationState | AnalystResult":
        context = self._retriever.search(config.client_id, request.text)

        # Deep dive confirmation flow
        if (
            self._analyst_agent is not None
            and SkillModule.DEEP_ANALYSIS in config.enabled_skills
            and clarification_state is not None
            and clarification_state.deep_dive_pending
            and (
                clarification_state.deep_dive_confirmed
                or any(
                    a.strip().lower() in ("yes", "y", "sure", "go ahead", "ok", "yep")
                    for a in clarification_state.answers_received
                )
            )
        ):
            checkpoints: list[str] = []
            return self._analyst_agent.run_deep(
                request=request,
                config=config,
                on_checkpoint=lambda step, thought: checkpoints.append(f"Step {step}: {thought}"),
            )

        state = self._clarifier.check(request, context, clarification_state)

        if not state.is_resolved:
            return state

        # Deep intent detection — only on first call (no existing clarification_state)
        if (
            self._analyst_agent is not None
            and SkillModule.DEEP_ANALYSIS in config.enabled_skills
            and clarification_state is None
            and self._detect_deep_intent(request)
        ):
            deep_state = ClarificationState(original_request=request)
            deep_state.questions_asked = [
                "Want me to do a deep dive on this? It may take a few minutes. "
                "Reply yes to proceed or no for a quick answer."
            ]
            deep_state.deep_dive_pending = True
            deep_state.is_resolved = False
            return deep_state

        plan = self._plan(request, context)
        charts: list[bytes] = []
        sql_data: dict | None = None
        sql_queries: list[str] = []
        anomalies: list[dict] = []

        # Spreadsheet routing — takes priority over SQL if file attached
        if (
            request.file_bytes is not None
            and request.filename is not None
            and self._spreadsheet_agent is not None
            and SkillModule.SPREADSHEET_ANALYSIS in config.enabled_skills
        ):
            sheet_result = self._spreadsheet_agent.run(
                file_bytes=request.file_bytes,
                filename=request.filename,
                question=request.text,
                config=config,
            )
            if sheet_result.success:
                sql_data = sheet_result.data

        # Funnel: look up stages in KB first; clarify if not defined
        if (
            plan.get("funnel")
            and SkillModule.FUNNEL_ANALYSIS in config.enabled_skills
        ):
            funnel_docs = self._retriever.search(
                config.client_id, "funnel stages conversion steps flow"
            )
            if funnel_docs:
                context = context + funnel_docs[:2]
            else:
                funnel_state = ClarificationState(original_request=request)
                funnel_state.questions_asked = [
                    "I don't have your funnel stages defined. What are the conversion steps? "
                    "(e.g., Visit → Signup → Active User → Paid Customer)"
                ]
                funnel_state.is_resolved = False
                return funnel_state

        if plan.get("sql") and SkillModule.SQL_QUERYING in config.enabled_skills:
            analysis_mode = None
            if plan.get("funnel") and SkillModule.FUNNEL_ANALYSIS in config.enabled_skills:
                analysis_mode = "funnel"
            elif plan.get("cohort") and SkillModule.COHORT_ANALYSIS in config.enabled_skills:
                analysis_mode = "cohort"
            sql_result = self._sql_agent.run(
                config.client_id, request.text, context, analysis_mode=analysis_mode
            )
            if sql_result.success:
                sql_data = sql_result.data
                if sql_data and sql_data.get("query"):
                    sql_queries.append(sql_data["query"])

                if plan.get("chart") and SkillModule.DATA_VISUALIZATION in config.enabled_skills and sql_data:
                    chart_result = self._chart_agent.run(sql_data)
                    if chart_result.success and chart_result.chart_png:
                        charts.append(chart_result.chart_png)

        # Anomaly detection — runs on sql_data when skill enabled
        anomaly_skill_enabled = (
            SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
            or SkillModule.STATISTICAL_ANOMALY in config.enabled_skills
        )
        if self._anomaly_agent is not None and anomaly_skill_enabled and sql_data:
            if (SkillModule.HARD_RULE_ANOMALY in config.enabled_skills
                    and SkillModule.STATISTICAL_ANOMALY in config.enabled_skills):
                mode = "both"
            elif SkillModule.HARD_RULE_ANOMALY in config.enabled_skills:
                mode = "hard"
            else:
                mode = "statistical"
            anomaly_result = self._anomaly_agent.run(config.client_id, sql_data, mode=mode)
            if anomaly_result.success:
                anomalies = anomaly_result.data.get("anomalies", [])

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

        # Hypothesis testing — runs on sql_data when skill enabled
        if (
            plan.get("hypothesis")
            and self._hypothesis_agent is not None
            and SkillModule.HYPOTHESIS_TESTING in config.enabled_skills
            and sql_data
        ):
            hyp_result = self._hypothesis_agent.run(config.client_id, request.text, sql_data)
            if hyp_result.success:
                sql_data = {**(sql_data or {}), **hyp_result.data}

        # Segmentation — runs on sql_data when skill enabled
        if (
            plan.get("segment")
            and self._segmentation_agent is not None
            and SkillModule.SEGMENTATION in config.enabled_skills
            and sql_data
        ):
            seg_result = self._segmentation_agent.run(config.client_id, request.text, sql_data)
            if seg_result.success:
                if seg_result.chart_png:
                    charts.append(seg_result.chart_png)
                sql_data = {**(sql_data or {}), **seg_result.data}

        # A/B testing — runs on sql_data when skill enabled
        if (
            plan.get("ab_test")
            and self._ab_agent is not None
            and SkillModule.AB_TESTING in config.enabled_skills
            and sql_data
        ):
            ab_result = self._ab_agent.run(config.client_id, request.text, sql_data)
            if ab_result.success:
                sql_data = {**(sql_data or {}), **ab_result.data}

        text = self._generate_response(request, context, sql_data, state.assumptions)

        # Deck Agent — builds PPTX via storyline (LLM) or fallback flat sections
        deck_pptx: bytes | None = None
        if (
            plan.get("deck")
            and self._deck_agent is not None
            and SkillModule.PRESENTATION_BUILDING in config.enabled_skills
        ):
            findings = (
                [f"{r}" for r in (sql_data.get("rows", []) or [])[:5]]
                if sql_data else []
            )
            storyline = self._deck_agent.build_storyline(
                analysis_text=text,
                findings=findings,
                solutions=[],
                recommendation="",
            )
            deck_result = self._deck_agent.run(
                title=request.text[:100], storyline=storyline, charts=charts
            )
            if deck_result.success:
                deck_pptx = deck_result.deck_pptx

        response = Response(
            request_id=str(uuid.uuid4()),
            text=text,
            charts=charts,
            assumptions=state.assumptions,
            deck_pptx=deck_pptx,
            anomalies=[Anomaly(**a) for a in anomalies],
        )

        # Interaction memory — always log when logger configured
        if self._memory_logger is not None:
            self._memory_logger.log(
                request=request,
                response=response,
                sql_queries=sql_queries,
            )

        return response

    def _plan(self, request: Request, context: list[str]) -> dict:
        system = (
            "Determine which capabilities are needed to answer this data request. "
            "Return JSON only: "
            '{"sql": true/false, "chart": true/false, "ml": true/false, '
            '"deck": true/false, "funnel": true/false, "cohort": true/false, '
            '"hypothesis": true/false, "segment": true/false, "ab_test": true/false}. '
            "Set ml=true for forecast/predict/projection/trend requests. "
            "Set deck=true for slide/deck/presentation/powerpoint requests. "
            "Set funnel=true for funnel or conversion analysis requests. "
            "Set cohort=true for cohort or retention analysis requests. "
            "Set hypothesis=true for hypothesis test/statistical significance/p-value/compare groups requests. "
            "Set segment=true for customer segmentation/clustering/group customers requests. "
            "Set ab_test=true for A/B test/experiment/variant evaluation requests."
        )
        user = f"Request: {request.text}\nContext: {chr(10).join(context)}"
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"sql": True, "chart": True, "ml": False, "deck": False,
                "funnel": False, "cohort": False,
                "hypothesis": False, "segment": False, "ab_test": False}

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
