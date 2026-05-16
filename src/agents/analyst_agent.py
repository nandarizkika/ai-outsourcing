# src/agents/analyst_agent.py
import json
import logging
from typing import Callable

import numpy as np
from scipy import stats

from src.core.llm import LLMRouter
from src.core.models import (
    AgentResult, AnalystResult, ClientConfig, Request, StepRecord, TaskType,
)
from src.knowledge.retriever import KnowledgeRetriever

_logger = logging.getLogger(__name__)

_MAX_STEPS_HARD_CAP = 15
_TOOL_FAILURE_LIMIT = 3


class AnalystAgent:
    def __init__(
        self,
        llm: LLMRouter,
        sql_agent,
        retriever: KnowledgeRetriever,
        ml_agent=None,
    ) -> None:
        self._llm = llm
        self._sql_agent = sql_agent
        self._retriever = retriever
        self._ml_agent = ml_agent

    def run_deep(
        self,
        request: Request,
        config: ClientConfig,
        on_checkpoint: Callable[[int, str], None],
        max_steps: int = 10,
    ) -> AnalystResult:
        max_steps = min(max_steps, _MAX_STEPS_HARD_CAP)
        history: list[StepRecord] = []
        consecutive_failures = 0

        for step in range(1, max_steps + 1):
            raw = self._think(request, history)
            try:
                parsed = json.loads(raw)
                thought = parsed.get("thought", "")
                tool = parsed.get("tool", "DONE")
                tool_input = parsed.get("tool_input", {})
            except json.JSONDecodeError:
                thought, tool, tool_input = "", "DONE", {}

            if tool == "DONE":
                break

            observation = self._execute_tool(tool, tool_input, config)
            if observation.startswith("Tool failed:"):
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            history.append(StepRecord(
                step=step,
                thought=thought,
                tool=tool,
                tool_input=tool_input,
                observation=observation,
            ))

            if step % 3 == 0:
                on_checkpoint(step, thought)

            if consecutive_failures >= _TOOL_FAILURE_LIMIT:
                _logger.warning("AnalystAgent: %d consecutive tool failures, stopping early", _TOOL_FAILURE_LIMIT)
                break

        return self._synthesize(request, history)

    def _think(self, request: Request, history: list[StepRecord]) -> str:
        tools_desc = (
            "Available tools: sql_query(question), stat_test(data, test, groups), "
            "cluster_segment(data, n_clusters), knowledge_search(query). "
            "When investigation is complete, return tool='DONE'."
        )
        system = (
            "You are an expert data analyst performing a root-cause investigation. "
            "At each step, decide what to investigate next. "
            f"{tools_desc} "
            "Return ONLY valid JSON: "
            '{"thought": "your reasoning", "tool": "tool_name", "tool_input": {...}}'
        )
        history_text = "\n".join(
            f"Step {r.step}: [{r.tool}] -> {r.observation[:300]}" for r in history
        )
        user = (
            f"Original question: {request.text}\n\n"
            f"Investigation history:\n{history_text}\n\n"
            "What should we investigate next?"
        )
        return self._llm.complete(TaskType.REASONING, system, user)

    def _execute_tool(self, tool: str, tool_input: dict, config: ClientConfig) -> str:
        try:
            if tool == "sql_query":
                question = tool_input.get("question", "")
                result = self._sql_agent.run(config.client_id, question, [])
                if result.success and result.data:
                    return json.dumps(result.data.get("rows", [])[:20])
                return f"Tool failed: {result.error}"

            if tool == "stat_test":
                return self._run_stat_test(tool_input)

            if tool == "cluster_segment":
                return self._run_cluster_segment(tool_input)

            if tool == "knowledge_search":
                query = tool_input.get("query", "")
                docs = self._retriever.search(config.client_id, query)
                return "\n".join(docs[:3]) if docs else "No documents found."

            return f"Tool failed: unknown tool '{tool}'"
        except Exception as exc:
            _logger.warning("Tool %s failed: %s", tool, exc)
            return f"Tool failed: {exc}"

    def _run_stat_test(self, tool_input: dict) -> str:
        test = tool_input.get("test", "ttest")
        groups = tool_input.get("groups", [])
        if len(groups) < 2:
            return "Tool failed: stat_test requires at least 2 groups"
        try:
            arrays = [np.array(g, dtype=float) for g in groups]
            if test == "ttest":
                stat, p = stats.ttest_ind(arrays[0], arrays[1])
            elif test == "anova":
                stat, p = stats.f_oneway(*arrays)
            elif test == "chi2":
                stat, p = stats.chisquare(arrays[0])
            else:
                return f"Tool failed: unknown test '{test}'"
            return json.dumps({"statistic": float(stat), "p_value": float(p), "significant": p < 0.05})
        except Exception as exc:
            return f"Tool failed: {exc}"

    def _run_cluster_segment(self, tool_input: dict) -> str:
        from sklearn.cluster import KMeans
        data = tool_input.get("data", [])
        n_clusters = int(tool_input.get("n_clusters", 3))
        if not data:
            return "Tool failed: no data provided"
        try:
            arr = np.array([[v for v in row.values() if isinstance(v, (int, float))]
                            for row in data], dtype=float)
            if arr.shape[0] < n_clusters:
                return "Tool failed: more clusters than data points"
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(arr)
            counts = {int(i): int((labels == i).sum()) for i in range(n_clusters)}
            return json.dumps({"n_clusters": n_clusters, "cluster_sizes": counts})
        except Exception as exc:
            return f"Tool failed: {exc}"

    def _synthesize(self, request: Request, history: list[StepRecord]) -> AnalystResult:
        history_text = "\n".join(
            f"Step {r.step}: [{r.tool}({json.dumps(r.tool_input)})] -> {r.observation[:400]}"
            for r in history
        )
        system = (
            "You are a senior analyst. Based on the investigation below, synthesize the findings. "
            "Return ONLY valid JSON: "
            '{"findings": ["str"], '
            '"solutions": [{"title": "str", "description": "str", "pros": ["str"], "cons": ["str"]}], '
            '"recommendation": "str"}'
        )
        user = (
            f"Original question: {request.text}\n\n"
            f"Investigation log:\n{history_text}\n\n"
            "Synthesize findings, propose solutions, and give a recommendation."
        )
        raw = self._llm.complete(TaskType.REASONING, system, user)
        try:
            parsed = json.loads(raw)
            return AnalystResult(
                agent_name="analyst_agent",
                success=True,
                steps=history,
                findings=parsed.get("findings", []),
                solutions=parsed.get("solutions", []),
                recommendation=parsed.get("recommendation", ""),
            )
        except json.JSONDecodeError:
            return AnalystResult(
                agent_name="analyst_agent",
                success=True,
                steps=history,
                findings=[],
                solutions=[],
                recommendation=raw[:500],
            )
