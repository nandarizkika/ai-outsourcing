import json

from src.core.llm import LLMRouter
from src.core.models import TaskType, Request, ClarificationState

MAX_ROUNDS = 2


class ClarificationChecker:
    def __init__(self, llm: LLMRouter):
        self._llm = llm

    def check(
        self,
        request: Request,
        context: list[str],
        state: ClarificationState | None = None,
    ) -> ClarificationState:
        if state is None:
            state = ClarificationState(original_request=request)

        if state.rounds >= MAX_ROUNDS:
            assumptions = self._generate_assumptions(request, state, context)
            return ClarificationState(
                original_request=request,
                rounds=state.rounds,
                questions_asked=state.questions_asked,
                answers_received=state.answers_received,
                is_resolved=True,
                assumptions=assumptions,
            )

        system = (
            "You are an AI data analyst assessing if a request is specific enough to execute. "
            "A request needs: what metric/data, what time period (if relevant), what filters (if relevant). "
            'Return JSON only: {"is_clear": true/false, "questions": ["q1"] | []}. '
            "If clear, questions must be empty. If not, provide 1-2 specific questions only."
        )
        user = (
            f"Request: {request.text}\n\n"
            f"Business context:\n{chr(10).join(context)}\n\n"
            "Is this specific enough to execute?"
        )

        raw = self._llm.complete(TaskType.SIMPLE, system, user)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            state.is_resolved = True
            return state

        if result.get("is_clear"):
            state.is_resolved = True
        else:
            state.questions_asked.extend(result.get("questions", []))
            state.rounds += 1

        return state

    def _generate_assumptions(
        self, request: Request, state: ClarificationState, context: list[str]
    ) -> list[str]:
        system = (
            "You are an AI data analyst. State the assumptions you will use to proceed with an ambiguous request. "
            "Return a JSON list of strings."
        )
        user = (
            f"Request: {request.text}\n"
            f"Questions asked: {state.questions_asked}\n"
            f"Answers received: {state.answers_received}\n"
            f"Context: {chr(10).join(context)}"
        )
        raw = self._llm.complete(TaskType.SIMPLE, system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ["Proceeding with best interpretation of the original request."]
