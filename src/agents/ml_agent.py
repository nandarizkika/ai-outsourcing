"""
MLAgent — multi-mode machine-learning agent.

Supported tasks (via `task` parameter or keyword auto-detection):
  * forecast   — time-series forecasting (ARIMA → ETS → linear trend)
  * regression — OLS linear or polynomial regression
  * classification — logistic regression (scipy) with accuracy reporting
  * anomaly    — outlier detection (Z-score and IQR)

All modes return an AgentResult with:
  - chart_png  (PNG bytes)
  - data       (task-specific dict including LLM interpretation)
  - error      (on failure)
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.models import AgentResult, TaskType

_logger = logging.getLogger(__name__)

# Minimum row counts per task
_MIN_FORECAST = 5
_MIN_REGRESSION = 3
_MIN_CLASSIFICATION = 4

# ---------------------------------------------------------------------------
# Keyword sets for auto-detection
# ---------------------------------------------------------------------------
_FORECAST_KEYWORDS = {"forecast", "predict next", "future", "trend", "projection", "time series"}
_REGRESSION_KEYWORDS = {"regress", "linear model", "predict sales", "predict revenue", "correlat"}
_ANOMALY_KEYWORDS = {"anomaly", "anomalies", "outlier", "outliers", "detect", "spike", "unusual"}
_CLASSIFICATION_KEYWORDS = {"classif", "churn", "segment", "label", "categor"}


def _detect_task(request: str) -> str:
    req = request.lower()
    for kw in _ANOMALY_KEYWORDS:
        if kw in req:
            return "anomaly"
    for kw in _CLASSIFICATION_KEYWORDS:
        if kw in req:
            return "classification"
    for kw in _REGRESSION_KEYWORDS:
        if kw in req:
            return "regression"
    for kw in _FORECAST_KEYWORDS:
        if kw in req:
            return "forecast"
    return "forecast"  # default


class MLAgent:
    """Multi-mode ML agent: forecasting, regression, classification, anomaly detection."""

    def __init__(self, llm) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        client_id: str,
        request: str,
        data: dict,
        task: Optional[str] = None,
        model_hint: Optional[str] = None,
    ) -> AgentResult:
        """
        Parameters
        ----------
        client_id   : client identifier (unused internally, kept for interface consistency)
        request     : natural-language request string
        data        : dict with ``columns`` and ``rows`` keys
        task        : one of "forecast", "regression", "classification", "anomaly".
                      If omitted, auto-detected from *request*.
        model_hint  : optional sub-model override (e.g. "arima", "ets", "linear_trend",
                      "polynomial", "zscore", "iqr").
        """
        rows = data.get("rows", [])
        columns = data.get("columns", [])
        if not rows or not columns:
            return AgentResult(
                agent_name="ml_agent", success=False, error="No data provided"
            )

        resolved_task = task or _detect_task(request)

        dispatch = {
            "forecast": self._run_forecast,
            "regression": self._run_regression,
            "classification": self._run_classification,
            "anomaly": self._run_anomaly,
        }
        handler = dispatch.get(resolved_task, self._run_forecast)
        return handler(request=request, data=data, model_hint=model_hint)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def _run_forecast(self, request: str, data: dict, model_hint: Optional[str]) -> AgentResult:
        df = pd.DataFrame(data["rows"], columns=data["columns"])
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error="No numeric column found for forecasting",
            )

        target_col = numeric_cols[-1]
        series = df[target_col].dropna().values.astype(float)

        if len(series) < _MIN_FORECAST:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error=f"Insufficient data: need at least {_MIN_FORECAST} points, got {len(series)}",
            )

        steps = max(1, min(len(series) // 2, 7))
        hint = (model_hint or "").lower()

        forecast_values, model_name = self._forecast_with_model(series, steps, hint)

        chart_png = self._render_forecast(series, forecast_values, target_col, model_name)
        interpretation = self._interpret(
            request=request,
            context=(
                f"Column: {target_col}\n"
                f"Model: {model_name}\n"
                f"Last {min(5, len(series))} historical values: {series[-5:].tolist()}\n"
                f"Next {len(forecast_values)} forecast values: {forecast_values.tolist()}"
            ),
        )

        return AgentResult(
            agent_name="ml_agent",
            success=True,
            chart_png=chart_png,
            data={
                "task": "forecast",
                "model": model_name,
                "target_column": target_col,
                "forecast": forecast_values.tolist(),
                "interpretation": interpretation,
            },
        )

    def _forecast_with_model(
        self, series: np.ndarray, steps: int, hint: str
    ) -> tuple[np.ndarray, str]:
        if hint == "linear_trend":
            return self._linear_trend_forecast(series, steps), "LinearTrend"
        if hint == "ets":
            try:
                return self._ets_forecast(series, steps), "ETS"
            except Exception as exc:
                _logger.warning("ETS failed (%s), falling back to linear trend", exc)
                return self._linear_trend_forecast(series, steps), "LinearTrend"

        # Default: try ARIMA → ETS → LinearTrend
        try:
            return self._arima_forecast(series, steps), "ARIMA"
        except Exception as exc:
            _logger.warning("ARIMA failed (%s), trying ETS", exc)

        try:
            return self._ets_forecast(series, steps), "ETS"
        except Exception as exc:
            _logger.warning("ETS failed (%s), falling back to linear trend", exc)
            return self._linear_trend_forecast(series, steps), "LinearTrend"

    def _arima_forecast(self, series: np.ndarray, steps: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        fit = model.fit()
        return np.asarray(fit.forecast(steps=steps))

    def _ets_forecast(self, series: np.ndarray, steps: int) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(series, trend="add", seasonal=None)
        fit = model.fit(optimized=True)
        return np.asarray(fit.forecast(steps))

    def _linear_trend_forecast(self, series: np.ndarray, steps: int) -> np.ndarray:
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series) + steps)
        return np.polyval(coeffs, future_x)

    def _render_forecast(
        self,
        historical: np.ndarray,
        forecast: np.ndarray,
        col_name: str,
        model_name: str,
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 5))
        hist_x = np.arange(len(historical))
        fore_x = np.arange(len(historical), len(historical) + len(forecast))
        ax.plot(hist_x, historical, marker="o", label="Historical", color="steelblue")
        ax.plot(fore_x, forecast, marker="o", linestyle="--", label=f"Forecast ({model_name})", color="orange")
        ax.axvline(x=len(historical) - 1, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Period")
        ax.set_ylabel(col_name)
        ax.set_title(f"{col_name} Forecast — {model_name}")
        ax.legend()
        plt.tight_layout()
        return self._fig_to_png(fig)

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def _run_regression(self, request: str, data: dict, model_hint: Optional[str]) -> AgentResult:
        df = pd.DataFrame(data["rows"], columns=data["columns"])
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error="Regression requires at least 2 numeric columns (feature + target)",
            )

        x_col, y_col = numeric_cols[0], numeric_cols[-1]
        x = df[x_col].dropna().values.astype(float)
        y = df[y_col].dropna().values.astype(float)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        if n < _MIN_REGRESSION:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error=f"Insufficient data: need at least {_MIN_REGRESSION} rows for regression, got {n}",
            )

        hint = (model_hint or "").lower()
        if hint == "polynomial":
            degree = 2
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            model_name = f"Polynomial(degree={degree})"
        else:
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            model_name = "OLS Linear"

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        chart_png = self._render_regression(x, y, y_pred, x_col, y_col, model_name)
        interpretation = self._interpret(
            request=request,
            context=(
                f"Model: {model_name}\n"
                f"Feature: {x_col}, Target: {y_col}\n"
                f"Coefficients: {coeffs.tolist()}\n"
                f"R²: {r_squared:.4f}"
            ),
        )

        return AgentResult(
            agent_name="ml_agent",
            success=True,
            chart_png=chart_png,
            data={
                "task": "regression",
                "model": model_name,
                "feature_column": x_col,
                "target_column": y_col,
                "coefficients": coeffs.tolist(),
                "r_squared": round(r_squared, 4),
                "interpretation": interpretation,
            },
        )

    def _render_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        x_col: str,
        y_col: str,
        model_name: str,
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(9, 5))
        sort_idx = np.argsort(x)
        ax.scatter(x, y, color="steelblue", label="Actual", zorder=3)
        ax.plot(x[sort_idx], y_pred[sort_idx], color="orange", linewidth=2, label=f"Fit ({model_name})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} ~ {x_col}  [{model_name}]")
        ax.legend()
        plt.tight_layout()
        return self._fig_to_png(fig)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _run_classification(self, request: str, data: dict, model_hint: Optional[str]) -> AgentResult:
        df = pd.DataFrame(data["rows"], columns=data["columns"])
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error="Classification requires at least 2 numeric columns (features + binary target)",
            )

        target_col = numeric_cols[-1]
        feature_cols = numeric_cols[:-1]
        df_clean = df[numeric_cols].dropna()

        if len(df_clean) < _MIN_CLASSIFICATION:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error=f"Insufficient data: need at least {_MIN_CLASSIFICATION} rows, got {len(df_clean)}",
            )

        X = df_clean[feature_cols].values.astype(float)
        y = df_clean[target_col].values.astype(float)

        # Normalize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X_mean) / X_std

        weights, accuracy = self._logistic_regression_fit(X_norm, y)
        unique_classes = np.unique(y)

        chart_png = self._render_classification(X_norm, y, weights, feature_cols, target_col)
        interpretation = self._interpret(
            request=request,
            context=(
                f"Task: binary classification\n"
                f"Target: {target_col}\n"
                f"Features: {feature_cols}\n"
                f"Classes: {unique_classes.tolist()}\n"
                f"Training accuracy: {accuracy:.2%}"
            ),
        )

        return AgentResult(
            agent_name="ml_agent",
            success=True,
            chart_png=chart_png,
            data={
                "task": "classification",
                "model": "LogisticRegression",
                "target_column": target_col,
                "feature_columns": feature_cols,
                "accuracy": round(accuracy, 4),
                "classes": unique_classes.tolist(),
                "interpretation": interpretation,
            },
        )

    def _logistic_regression_fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, iterations: int = 300
    ) -> tuple[np.ndarray, float]:
        """Gradient-descent logistic regression (pure numpy)."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features + 1)  # includes bias

        X_b = np.column_stack([np.ones(n_samples), X])

        for _ in range(iterations):
            z = X_b @ w
            sigmoid = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            error = sigmoid - y
            gradient = X_b.T @ error / n_samples
            w -= lr * gradient

        z_final = X_b @ w
        probs = 1.0 / (1.0 + np.exp(-np.clip(z_final, -500, 500)))
        preds = (probs >= 0.5).astype(float)
        accuracy = float(np.mean(preds == y))
        return w, accuracy

    def _render_classification(
        self,
        X_norm: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_cols: list,
        target_col: str,
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(8, 5))
        classes = np.unique(y)
        colors = ["steelblue", "orange", "green", "red"]

        if X_norm.shape[1] >= 2:
            for i, cls in enumerate(classes):
                mask = y == cls
                ax.scatter(
                    X_norm[mask, 0],
                    X_norm[mask, 1],
                    label=f"{target_col}={int(cls)}",
                    color=colors[i % len(colors)],
                    alpha=0.7,
                )
            ax.set_xlabel(f"{feature_cols[0]} (normalized)")
            ax.set_ylabel(f"{feature_cols[1]} (normalized)")
        else:
            for i, cls in enumerate(classes):
                mask = y == cls
                ax.scatter(
                    X_norm[mask, 0],
                    np.zeros(mask.sum()),
                    label=f"{target_col}={int(cls)}",
                    color=colors[i % len(colors)],
                    alpha=0.7,
                )
            ax.set_xlabel(f"{feature_cols[0]} (normalized)")

        ax.set_title(f"Classification: {target_col} (Logistic Regression)")
        ax.legend()
        plt.tight_layout()
        return self._fig_to_png(fig)

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------

    def _run_anomaly(self, request: str, data: dict, model_hint: Optional[str]) -> AgentResult:
        df = pd.DataFrame(data["rows"], columns=data["columns"])
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error="No numeric column found for anomaly detection",
            )

        target_col = numeric_cols[-1]
        series = df[target_col].dropna().values.astype(float)

        if len(series) < 3:
            return AgentResult(
                agent_name="ml_agent",
                success=False,
                error=f"Insufficient data: need at least 3 points, got {len(series)}",
            )

        hint = (model_hint or "").lower()
        if hint == "iqr":
            anomaly_indices, method_name = self._iqr_anomalies(series)
        else:
            # Default: Z-score; fall back to IQR if std ~ 0
            anomaly_indices, method_name = self._zscore_anomalies(series)
            if not anomaly_indices:
                iqr_indices, iqr_name = self._iqr_anomalies(series)
                if iqr_indices:
                    anomaly_indices, method_name = iqr_indices, iqr_name

        anomaly_values = [
            {"index": int(i), "value": float(series[i])} for i in anomaly_indices
        ]

        chart_png = self._render_anomaly(series, anomaly_indices, target_col, method_name)
        interpretation = self._interpret(
            request=request,
            context=(
                f"Column: {target_col}\n"
                f"Method: {method_name}\n"
                f"Total points: {len(series)}\n"
                f"Anomalies found: {len(anomaly_values)} at indices {anomaly_indices}"
            ),
        )

        return AgentResult(
            agent_name="ml_agent",
            success=True,
            chart_png=chart_png,
            data={
                "task": "anomaly",
                "model": method_name,
                "target_column": target_col,
                "anomalies": anomaly_values,
                "interpretation": interpretation,
            },
        )

    def _zscore_anomalies(self, series: np.ndarray, threshold: float = 2.5) -> tuple[list[int], str]:
        mean = np.mean(series)
        std = np.std(series)
        if std < 1e-10:
            return [], "ZScore"
        z_scores = np.abs((series - mean) / std)
        indices = [int(i) for i in np.where(z_scores > threshold)[0]]
        return indices, "ZScore"

    def _iqr_anomalies(self, series: np.ndarray, factor: float = 1.5) -> tuple[list[int], str]:
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        indices = [int(i) for i in np.where((series < lower) | (series > upper))[0]]
        return indices, "IQR"

    def _render_anomaly(
        self,
        series: np.ndarray,
        anomaly_indices: list[int],
        col_name: str,
        method_name: str,
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(series))
        ax.plot(x, series, marker="o", color="steelblue", label="Values")
        if anomaly_indices:
            ax.scatter(
                anomaly_indices,
                series[anomaly_indices],
                color="red",
                zorder=5,
                s=100,
                label="Anomaly",
            )
        ax.set_xlabel("Index")
        ax.set_ylabel(col_name)
        ax.set_title(f"Anomaly Detection — {col_name} ({method_name})")
        ax.legend()
        plt.tight_layout()
        return self._fig_to_png(fig)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _interpret(self, request: str, context: str) -> str:
        system = (
            "You are a senior data analyst. Given model results, write a concise 2-3 sentence "
            "plain-language interpretation that directly addresses the user's request."
        )
        user = f"Request: {request}\n\nModel results:\n{context}\n\nInterpretation:"
        return self._llm.complete(TaskType.REASONING, system, user)

    @staticmethod
    def _fig_to_png(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
