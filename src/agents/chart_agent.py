import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.core.models import AgentResult


class ChartAgent:
    def run(self, data: dict) -> AgentResult:
        if not data.get("rows") or not data.get("columns"):
            return AgentResult(agent_name="chart_agent", success=False, error="No data to chart")

        df = pd.DataFrame(data["rows"], columns=data["columns"])

        if df.empty:
            return AgentResult(agent_name="chart_agent", success=False, error="Empty dataset")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        # Single value — no chart needed
        if len(df) == 1 and len(numeric_cols) <= 1 and not non_numeric_cols:
            return AgentResult(agent_name="chart_agent", success=True, data={"skipped": True})

        png = self._render(df, numeric_cols, non_numeric_cols)
        if png is None:
            return AgentResult(agent_name="chart_agent", success=True, data={"skipped": True})

        return AgentResult(agent_name="chart_agent", success=True, chart_png=png)

    def _render(self, df: pd.DataFrame, numeric_cols: list, non_numeric_cols: list) -> Optional[bytes]:
        fig, ax = plt.subplots(figsize=(10, 6))

        if non_numeric_cols and numeric_cols:
            x_col, y_col = non_numeric_cols[0], numeric_cols[0]
            ax.bar(df[x_col].astype(str), df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{y_col} by {x_col}")
        elif len(numeric_cols) >= 2:
            ax.plot(df[numeric_cols[0]], df[numeric_cols[1]], marker="o")
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"{numeric_cols[1]} over {numeric_cols[0]}")
        else:
            plt.close(fig)
            return None

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
