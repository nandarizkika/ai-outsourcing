import io
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)


class SegmentationAgent:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def run(self, client_id: str, request: str, data: dict) -> AgentResult:
        try:
            df = pd.DataFrame(data.get("rows", []))
            num_df = df.select_dtypes(include=[np.number])

            if num_df.shape[1] < 2:
                return AgentResult(agent_name="segmentation_agent", success=False,
                                   error="Need at least 2 numeric columns for segmentation")
            if len(num_df) < 4:
                return AgentResult(agent_name="segmentation_agent", success=False,
                                   error="Need at least 4 rows for segmentation")

            X = StandardScaler().fit_transform(num_df.values)
            max_k = min(6, len(X) // 2)
            if max_k < 2:
                max_k = 2

            best_k, best_score = 2, -1.0
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
                if score > best_score:
                    best_score, best_k = score, k

            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            df = df.copy()
            df["segment"] = labels

            centroids = {
                f"segment_{seg}": {
                    col: float(num_df[labels == seg][col].mean())
                    for col in num_df.columns
                }
                for seg in range(best_k)
            }

            col1, col2 = num_df.columns[0], num_df.columns[1]
            fig, ax = plt.subplots(figsize=(8, 5))
            for seg in range(best_k):
                mask = labels == seg
                ax.scatter(num_df[mask][col1], num_df[mask][col2],
                           label=f"Segment {seg}", alpha=0.7)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f"Customer Segmentation (k={best_k})")
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            chart_png = buf.read()

            interpretation = ""
            if self._llm is not None:
                system = "Describe these customer segments in 1-2 sentences per segment for a business audience."
                user = f"Segments and their average metrics: {centroids}"
                interpretation = self._llm.complete(TaskType.SIMPLE, system, user)

            return AgentResult(
                agent_name="segmentation_agent",
                success=True,
                chart_png=chart_png,
                data={
                    "n_segments": best_k,
                    "silhouette_score": float(best_score),
                    "centroids": centroids,
                    "interpretation": interpretation,
                    "rows": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                },
            )
        except Exception as exc:
            _logger.error("SegmentationAgent failed: %s", exc, exc_info=True)
            return AgentResult(agent_name="segmentation_agent", success=False, error=str(exc))
