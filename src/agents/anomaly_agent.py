import json
import numpy as np

from src.core.models import AgentResult, Anomaly
from src.knowledge.retriever import KnowledgeRetriever


class AnomalyAgent:
    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    def run(
        self,
        client_id: str,
        data: dict,
        mode: str = "both",
    ) -> AgentResult:
        anomalies: list[Anomaly] = []
        if mode in ("hard", "both"):
            anomalies.extend(self._check_hard_rules(client_id, data))
        if mode in ("statistical", "both"):
            anomalies.extend(self._check_statistical(data))
        return AgentResult(
            agent_name="anomaly_agent",
            success=True,
            data={"anomalies": [a.model_dump() for a in anomalies]},
        )

    def _check_hard_rules(self, client_id: str, data: dict) -> list[Anomaly]:
        docs = self._retriever.search(client_id, "anomaly rules thresholds metrics")
        anomalies: list[Anomaly] = []
        for doc in docs:
            try:
                rule = json.loads(doc)
                if not all(k in rule for k in ("metric", "operator", "threshold")):
                    continue
                metric = rule["metric"]
                op = rule["operator"]
                threshold = float(rule["threshold"])
                severity = rule.get("severity", "warning")
                for row in data.get("rows", []):
                    if metric not in row:
                        continue
                    try:
                        value = float(row[metric])
                    except (TypeError, ValueError):
                        continue
                    breached = (
                        (op == ">" and value > threshold)
                        or (op == ">=" and value >= threshold)
                        or (op == "<" and value < threshold)
                        or (op == "<=" and value <= threshold)
                    )
                    if breached:
                        anomalies.append(Anomaly(
                            metric=metric,
                            value=value,
                            threshold=threshold,
                            operator=op,
                            severity=severity,
                            description=f"{metric} is {value} (rule: {metric} {op} {threshold})",
                            mode="hard_rule",
                        ))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        return anomalies

    def _check_statistical(self, data: dict) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        rows = data.get("rows", [])
        if len(rows) < 3:
            return anomalies
        for col in data.get("columns", []):
            values = [
                float(row[col])
                for row in rows
                if col in row and isinstance(row[col], (int, float))
            ]
            if len(values) < 3:
                continue
            arr = np.array(values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            if std == 0:
                continue
            for val in values:
                z = abs((val - mean) / std)
                if z > 2.5:
                    anomalies.append(Anomaly(
                        metric=col,
                        value=val,
                        expected=mean,
                        severity="critical" if z >= 3.0 else "warning",
                        description=(
                            f"{col} value {val:.2f} is {z:.1f} std devs "
                            f"from mean ({mean:.2f})"
                        ),
                        mode="statistical",
                    ))
        return anomalies
