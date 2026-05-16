import logging
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class ABTestingAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            df = pd.DataFrame(data.get("rows", []))

            variant_col = next(
                (c for c in df.columns
                 if pd.api.types.is_string_dtype(df[c])
                 and "control" in [str(v).lower() for v in df[c].unique()]),
                None,
            )
            if variant_col is None:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error="No variant column with 'control' group found")

            num_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c]) and c != variant_col]
            if not num_cols:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error="No numeric metric column found")

            metric_col = num_cols[0]
            groups = df[variant_col].unique()
            control_label = next(g for g in groups if str(g).lower() == "control")
            non_control = [g for g in groups if str(g).lower() != "control"]
            if len(non_control) != 1:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error=f"Expected exactly 1 non-control group, found {len(non_control)}: {list(non_control)}")
            # Check if the variant is also a control variant (e.g., control2)
            variant_label = non_control[0]
            if str(variant_label).lower().startswith("control"):
                return AgentResult(agent_name="ab_agent", success=False,
                                   error=f"Variant group '{variant_label}' appears to be a control variant, not a treatment group")

            control_data = df[df[variant_col] == control_label][metric_col].dropna().values
            variant_data = df[df[variant_col] == variant_label][metric_col].dropna().values

            if len(control_data) < 2 or len(variant_data) < 2:
                return AgentResult(agent_name="ab_agent", success=False,
                                   error="Each group needs at least 2 observations")

            unique_vals = set(df[metric_col].dropna().unique())
            if unique_vals <= {0, 1}:
                count_c, count_v = int(control_data.sum()), int(variant_data.sum())
                n_c, n_v = len(control_data), len(variant_data)
                stat, p_value = proportions_ztest([count_v, count_c], [n_v, n_c])
                test_name = "proportions_ztest"
                control_mean = count_c / n_c
                variant_mean = count_v / n_v
            else:
                stat, p_value = stats.ttest_ind(variant_data, control_data, equal_var=False)
                test_name = "t_test"
                control_mean = float(control_data.mean())
                variant_mean = float(variant_data.mean())

            uplift = (variant_mean - control_mean) / control_mean if control_mean != 0 else 0.0
            significant = bool(p_value < 0.05)

            recommendation = ""
            if self._llm is not None:
                system = "Give a 1-sentence recommendation on whether to ship the variant based on this A/B test."
                user = (
                    f"Control: {control_mean:.4f}, Variant: {variant_mean:.4f}, "
                    f"Uplift: {uplift:.2%}, p-value: {p_value:.4f}, Significant: {significant}"
                )
                recommendation = self._llm.complete(TaskType.SIMPLE, system, user)

            return AgentResult(
                agent_name="ab_agent",
                success=True,
                data={
                    "test": test_name,
                    "variant_col": variant_col,
                    "metric_col": metric_col,
                    "control_mean": control_mean,
                    "variant_mean": variant_mean,
                    "uplift": uplift,
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": significant,
                    "recommendation": recommendation,
                },
            )
        except Exception as exc:
            _logger.error("ABTestingAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="ab_agent", success=False, error=str(exc))
