# src/agents/spreadsheet_agent.py
import io
import json
import logging

import pandas as pd

from src.core.llm import LLMRouter
from src.core.models import AgentResult, ClientConfig, TaskType

_logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class SpreadsheetAgent:
    def __init__(self, llm: LLMRouter) -> None:
        self._llm = llm

    def run(
        self,
        file_bytes: bytes,
        filename: str,
        question: str,
        config: ClientConfig,
    ) -> AgentResult:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in _SUPPORTED_EXTENSIONS:
            return AgentResult(
                agent_name="spreadsheet_agent",
                success=False,
                error=f"Unsupported file format: {ext}. Supported: {', '.join(_SUPPORTED_EXTENSIONS)}",
            )
        try:
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            return AgentResult(agent_name="spreadsheet_agent", success=False, error=str(e))

        system = (
            "You are a data analyst. Given a question and a DataFrame with the columns shown, "
            "return a single JSON object describing the pandas operation to perform. "
            'Valid operations: "describe", "groupby", "filter", "head". '
            'For groupby: {"operation": "groupby", "by": "<col>", "agg": {"<col>": "<func>"}}. '
            'For filter: {"operation": "filter", "column": "<col>", "operator": ">|<|>=|<=|==", "value": <n>}. '
            'For describe or head: {"operation": "describe"} or {"operation": "head", "n": 10}. '
            "Return ONLY valid JSON."
        )
        user = (
            f"Question: {question}\n"
            f"Columns: {list(df.columns)}\n"
            f"Sample (first 3 rows):\n{df.head(3).to_json(orient='records')}"
        )
        raw = self._llm.complete(TaskType.TOOL, system, user).strip()

        try:
            ops = json.loads(raw)
        except json.JSONDecodeError:
            ops = {"operation": "describe"}

        result_df = self._execute(df, ops)
        return AgentResult(
            agent_name="spreadsheet_agent",
            success=True,
            data={
                "rows": result_df.to_dict(orient="records"),
                "columns": list(result_df.columns),
                "summary": ops,
            },
        )

    def _execute(self, df: pd.DataFrame, ops: dict) -> pd.DataFrame:
        op = ops.get("operation", "describe")
        try:
            if op == "groupby":
                return df.groupby(ops["by"]).agg(ops["agg"]).reset_index()
            if op == "filter":
                col, operator, value = ops["column"], ops["operator"], ops["value"]
                if operator == ">":
                    return df[df[col] > value]
                if operator == ">=":
                    return df[df[col] >= value]
                if operator == "<":
                    return df[df[col] < value]
                if operator == "<=":
                    return df[df[col] <= value]
                if operator == "==":
                    return df[df[col] == value]
            if op == "head":
                return df.head(ops.get("n", 10))
        except (KeyError, TypeError) as exc:
            _logger.warning("SpreadsheetAgent operation failed: %s", exc)
        return df.describe().reset_index()
