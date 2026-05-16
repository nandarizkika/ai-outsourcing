import logging
import pandas as pd
from scipy import stats
from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class HypothesisAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            df = pd.DataFrame(data.get("rows", []))
            if df.empty:
                return AgentResult(agent_name="hypothesis_agent", success=False,
                                   error="No data provided")

            cat_cols = [c for c in df.columns
                        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
            num_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])]

            group_col = next((c for c in cat_cols if df[c].nunique() == 2), None)
            metric_col = next((c for c in num_cols if c != group_col), None)

            if group_col is None or metric_col is None:
                return AgentResult(
                    agent_name="hypothesis_agent", success=False,
                    error="Need exactly one 2-group column and one numeric metric column",
                )

            groups = df[group_col].unique()
            g1 = df[df[group_col] == groups[0]][metric_col].dropna().values
            g2 = df[df[group_col] == groups[1]][metric_col].dropna().values

            if len(g1) < 2 or len(g2) < 2:
                return AgentResult(
                    agent_name="hypothesis_agent", success=False,
                    error="Each group needs at least 2 observations",
                )

            unique_vals = set(df[metric_col].dropna().unique())
            if unique_vals <= {0, 1}:
                ct = pd.crosstab(df[group_col], df[metric_col])
                stat, p_value, _, _ = stats.chi2_contingency(ct)
                test_name = "chi2"
            else:
                stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
                test_name = "t_test"

            significant = bool(p_value < 0.05)
            interpretation = ""
            if self._llm is not None:
                system = "Interpret this statistical test result in 1-2 sentences for a business audience."
                user = (
                    f"Test: {test_name}, Groups: {groups[0]} vs {groups[1]}, "
                    f"Metric: {metric_col}, p-value: {p_value:.4f}, Significant: {significant}"
                )
                interpretation = self._llm.complete(TaskType.SIMPLE, system, user)

            return AgentResult(
                agent_name="hypothesis_agent",
                success=True,
                data={
                    "test": test_name,
                    "group_col": group_col,
                    "metric_col": metric_col,
                    "groups": list(groups),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": significant,
                    "interpretation": interpretation,
                },
            )
        except Exception as exc:
            _logger.error("HypothesisAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="hypothesis_agent", success=False, error=str(exc))
