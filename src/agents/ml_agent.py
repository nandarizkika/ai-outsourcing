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
_NLP_KEYWORDS = {"classify text", "nlp", "sentiment", "text classification", "text classify"}
_BUILD_MODEL_KEYWORDS = {"build model", "train model", "fit model", "best model"}


def _detect_task(request: str) -> str:
    req = request.lower()
    for kw in _BUILD_MODEL_KEYWORDS:
        if kw in req:
            return "build_model"
    for kw in _ANOMALY_KEYWORDS:
        if kw in req:
            return "anomaly"
    for kw in _NLP_KEYWORDS:
        if kw in req:
            return "nlp"
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
        target_col: str | None = None,
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

        if resolved_task == "build_model":
            if not target_col:
                return AgentResult(
                    agent_name="ml_agent", success=False,
                    error="target_col is required for build_model task",
                )
            return self._run_build_model(client_id, request, data, target_col)

        if resolved_task == "nlp":
            return self._run_nlp(client_id, request, data, target_col)

        dispatch = {
            "forecast": self._run_forecast,
            "regression": self._run_regression,
            "classification": self._run_classification,
            "anomaly": self._run_anomaly,
        }
        handler = dispatch.get(resolved_task, self._run_forecast)
        return handler(request=request, data=data, model_hint=model_hint)

    # ------------------------------------------------------------------
    # Build Model (full ML suite: regression + classification + HPO)
    # ------------------------------------------------------------------

    def _run_build_model(self, client_id: str, request: str, data: dict, target_col: str) -> AgentResult:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import cross_val_score, GridSearchCV
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVR, SVC
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.metrics import r2_score, mean_squared_error, f1_score
        import xgboost as xgb
        import lightgbm as lgb

        df = pd.DataFrame(data["rows"])
        if target_col not in df.columns:
            return AgentResult(
                agent_name="ml_agent", success=False,
                error=f"Target column '{target_col}' not found in data",
            )

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).values
        y_raw = df[target_col].values

        # Detect task type: classification if target is int/bool with few unique values
        unique_vals = len(np.unique(y_raw))
        is_classification = unique_vals <= 20 and np.issubdtype(y_raw.dtype, np.integer)

        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            candidates = [
                ("LogisticRegression", LogisticRegression(max_iter=500), {"C": [0.1, 1.0]}),
                ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42), {"max_depth": [3, 5]}),
                ("GradientBoosting", GradientBoostingClassifier(n_estimators=50, random_state=42), {"max_depth": [2, 3]}),
                ("XGBoost", xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0, eval_metric="logloss"), {"max_depth": [2, 3]}),
                ("LightGBM", lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1), {"num_leaves": [15, 31]}),
                ("SVC", SVC(), {"C": [0.1, 1.0]}),
                ("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 5]}),
                ("GaussianNB", GaussianNB(), {}),
                ("MLP", MLPClassifier(max_iter=200, random_state=42), {"hidden_layer_sizes": [(50,), (100,)]}),
            ]
            scorer = "f1_weighted"
        else:
            y = y_raw.astype(float)
            candidates = [
                ("LinearRegression", LinearRegression(), {}),
                ("Ridge", Ridge(), {"alpha": [0.1, 1.0]}),
                ("Lasso", Lasso(max_iter=2000), {"alpha": [0.01, 0.1]}),
                ("RandomForest", RandomForestRegressor(n_estimators=50, random_state=42), {"max_depth": [3, 5]}),
                ("GradientBoosting", GradientBoostingRegressor(n_estimators=50, random_state=42), {"max_depth": [2, 3]}),
                ("XGBoost", xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0), {"max_depth": [2, 3]}),
                ("LightGBM", lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), {"num_leaves": [15, 31]}),
                ("SVR", SVR(), {"C": [0.1, 1.0]}),
                ("KNN", KNeighborsRegressor(), {"n_neighbors": [3, 5]}),
                ("MLPRegressor", MLPRegressor(max_iter=200, random_state=42), {"hidden_layer_sizes": [(50,), (100,)]}),
            ]
            scorer = "r2"

        # Cross-val all candidates, pick top 3
        cv_scores: list[tuple[float, str, object, dict]] = []
        for name, model, params in candidates:
            try:
                scores = cross_val_score(model, X, y, cv=min(5, len(y) // 5), scoring=scorer)
                cv_scores.append((float(scores.mean()), name, model, params))
            except Exception:
                continue

        if not cv_scores:
            return AgentResult(agent_name="ml_agent", success=False, error="All candidates failed cross-validation")

        cv_scores.sort(key=lambda t: t[0], reverse=True)
        top3 = cv_scores[:3]

        # GridSearchCV on top 3
        best_score = -1e9
        best_name = ""
        best_model_obj = None
        for score, name, model, params in top3:
            try:
                if params:
                    gs = GridSearchCV(model, params, cv=min(3, len(y) // 3), scoring=scorer)
                    gs.fit(X, y)
                    tuned_score = gs.best_score_
                    fitted = gs.best_estimator_
                else:
                    model.fit(X, y)
                    tuned_score = score
                    fitted = model
                if tuned_score > best_score:
                    best_score = tuned_score
                    best_name = name
                    best_model_obj = fitted
            except Exception:
                continue

        if best_model_obj is None:
            return AgentResult(agent_name="ml_agent", success=False, error="GridSearch failed for all top candidates")

        # Feature importance
        feature_importance: dict[str, float] = {}
        feature_names = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        if hasattr(best_model_obj, "feature_importances_"):
            fi = best_model_obj.feature_importances_
            feature_importance = {feature_names[i]: float(fi[i]) for i in range(min(len(fi), len(feature_names)))}
        elif hasattr(best_model_obj, "coef_"):
            coef = best_model_obj.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            feature_importance = {feature_names[i]: float(abs(coef[i])) for i in range(min(len(coef), len(feature_names)))}

        # Final metrics on training data
        y_pred = best_model_obj.predict(X)
        if is_classification:
            result_data = {
                "best_model": best_name,
                "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
                "feature_importance": feature_importance,
                "best_params": getattr(best_model_obj, "get_params", lambda: {})(),
            }
        else:
            result_data = {
                "best_model": best_name,
                "r2": float(r2_score(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(np.mean(np.abs(y - y_pred))),
                "feature_importance": feature_importance,
                "best_params": getattr(best_model_obj, "get_params", lambda: {})(),
            }
        return AgentResult(agent_name="ml_agent", success=True, data=result_data)

    # ------------------------------------------------------------------
    # NLP Classification
    # ------------------------------------------------------------------

    def _run_nlp(self, client_id: str, request: str, data: dict, target_col: str | None) -> AgentResult:
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score
        from sentence_transformers import SentenceTransformer

        df = pd.DataFrame(data["rows"])

        # Find text column (first string column that isn't the target)
        text_col = None
        for col in df.columns:
            if col != target_col and (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
                text_col = col
                break
        if text_col is None:
            return AgentResult(agent_name="ml_agent", success=False,
                               error="No text column found in data")

        texts = df[text_col].astype(str).tolist()

        if target_col and target_col in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(df[target_col].values)

            classifiers = [
                ("LogisticRegression", LogisticRegression(max_iter=500)),
                ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("SVC", SVC()),
            ]

            # TF-IDF vectorizer
            try:
                tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                X_tfidf = tfidf.fit_transform(texts).toarray()
            except ValueError as e:
                return AgentResult(agent_name="ml_agent", success=False,
                                   error=f"TF-IDF vectorization failed: {e}")

            # Sentence embeddings
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            X_embed = st_model.encode(texts)

            best_f1 = -1.0
            best_vec_name = ""
            best_clf_name = ""
            best_clf_obj = None
            best_X = X_tfidf

            for vec_name, X in [("tfidf", X_tfidf), ("sentence_transformer", X_embed)]:
                for clf_name, clf in classifiers:
                    try:
                        scores = cross_val_score(clf, X, y, cv=min(3, len(y) // 3), scoring="f1_weighted")
                        mean_f1 = float(scores.mean())
                        if mean_f1 > best_f1:
                            best_f1 = mean_f1
                            best_vec_name = vec_name
                            best_clf_name = clf_name
                            best_clf_obj = clf
                            best_X = X
                    except Exception:
                        continue

            if best_clf_obj is None:
                return AgentResult(agent_name="ml_agent", success=False,
                                   error="All NLP classifiers failed")

            best_clf_obj.fit(best_X, y)
            y_pred = best_clf_obj.predict(best_X)
            return AgentResult(
                agent_name="ml_agent",
                success=True,
                data={
                    "best_vectorizer": best_vec_name,
                    "best_classifier": best_clf_name,
                    "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
                    "text_col": text_col,
                    "target_col": target_col,
                },
            )

        # No target — return basic text stats
        return AgentResult(
            agent_name="ml_agent",
            success=True,
            data={
                "text_col": text_col,
                "sample_count": len(texts),
                "avg_length": float(np.mean([len(t.split()) for t in texts])),
            },
        )

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
