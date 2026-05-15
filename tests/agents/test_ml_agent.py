from unittest.mock import MagicMock
import pytest

from src.agents.ml_agent import MLAgent
from src.core.models import AgentResult


def _make_agent(llm_response="Forecast: revenue will grow 12% over the next 4 weeks."):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = llm_response
    return MLAgent(llm=mock_llm)


def _time_series_data():
    return {
        "columns": ["week", "revenue"],
        "rows": [
            {"week": 1, "revenue": 100},
            {"week": 2, "revenue": 110},
            {"week": 3, "revenue": 108},
            {"week": 4, "revenue": 115},
            {"week": 5, "revenue": 120},
            {"week": 6, "revenue": 118},
            {"week": 7, "revenue": 125},
            {"week": 8, "revenue": 130},
            {"week": 9, "revenue": 128},
            {"week": 10, "revenue": 135},
        ],
    }


def test_forecast_returns_chart_and_text():
    agent = _make_agent()
    result = agent.run(
        client_id="client1",
        request="Forecast next 4 weeks of revenue",
        data=_time_series_data(),
    )
    assert result.success is True
    assert result.chart_png is not None
    assert len(result.chart_png) > 0
    assert result.data is not None
    assert "forecast" in result.data


def test_forecast_interpretation_comes_from_llm():
    agent = _make_agent(llm_response="Revenue will increase by 15%.")
    result = agent.run(
        client_id="client1",
        request="Forecast revenue",
        data=_time_series_data(),
    )
    assert result.success is True
    assert result.data["interpretation"] == "Revenue will increase by 15%."


def test_insufficient_data_returns_error():
    agent = _make_agent()
    result = agent.run(
        client_id="client1",
        request="Forecast revenue",
        data={"columns": ["revenue"], "rows": [{"revenue": 100}]},
    )
    assert result.success is False
    assert result.error is not None
    assert "insufficient" in result.error.lower() or "not enough" in result.error.lower()


def test_no_numeric_column_returns_error():
    agent = _make_agent()
    result = agent.run(
        client_id="client1",
        request="Forecast",
        data={
            "columns": ["name", "category"],
            "rows": [{"name": "A", "category": "X"}, {"name": "B", "category": "Y"}],
        },
    )
    assert result.success is False
    assert result.error is not None


def _tabular_regression_data():
    import numpy as np
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n) * 0.1
    rows = [{"f1": float(X[i, 0]), "f2": float(X[i, 1]), "f3": float(X[i, 2]),
             "target": float(y[i])} for i in range(n)]
    return {"columns": ["f1", "f2", "f3", "target"], "rows": rows}


def _tabular_classification_data():
    import numpy as np
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rows = [{"f1": float(X[i, 0]), "f2": float(X[i, 1]), "f3": float(X[i, 2]),
             "label": int(y[i])} for i in range(n)]
    return {"columns": ["f1", "f2", "f3", "label"], "rows": rows}


def test_build_model_regression_returns_best_model_name():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="build regression model for target",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="target",
    )
    assert result.success is True
    assert "best_model" in result.data
    assert isinstance(result.data["best_model"], str)


def test_build_model_regression_returns_metrics():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="predict target",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="target",
    )
    assert result.success is True
    assert "r2" in result.data
    assert "rmse" in result.data
    assert result.data["r2"] > 0.5


def test_build_model_classification_returns_best_model_and_report():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify label",
        data=_tabular_classification_data(),
        task="build_model",
        target_col="label",
    )
    assert result.success is True
    assert "best_model" in result.data
    assert "f1_weighted" in result.data
    assert result.data["f1_weighted"] > 0.5


def test_build_model_returns_feature_importance():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="predict target",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="target",
    )
    assert result.success is True
    assert "feature_importance" in result.data
    fi = result.data["feature_importance"]
    assert isinstance(fi, dict)
    assert len(fi) > 0


def test_build_model_fails_gracefully_with_missing_target_col():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="predict nonexistent",
        data=_tabular_regression_data(),
        task="build_model",
        target_col="nonexistent_col",
    )
    assert result.success is False
    assert result.error is not None


def _nlp_classification_data():
    texts = [
        "The product is amazing and I love it", "Great service, very satisfied",
        "Excellent quality, highly recommend", "Best purchase I have made",
        "Good value for money", "Really happy with this",
        "Terrible product, broke immediately", "Awful experience, very disappointed",
        "Worst purchase ever, complete waste", "Poor quality, do not buy",
        "Very bad service", "Disappointed with the result",
    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    rows = [{"text": t, "label": l} for t, l in zip(texts, labels)]
    return {"columns": ["text", "label"], "rows": rows}


def test_nlp_classify_returns_best_pipeline():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify sentiment",
        data=_nlp_classification_data(),
        task="nlp",
        target_col="label",
    )
    assert result.success is True
    assert "best_vectorizer" in result.data
    assert "best_classifier" in result.data
    assert "f1_weighted" in result.data


def test_nlp_classify_f1_above_chance():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify text",
        data=_nlp_classification_data(),
        task="nlp",
        target_col="label",
    )
    assert result.success is True
    assert result.data["f1_weighted"] > 0.5


def test_nlp_missing_text_col_returns_failure():
    agent = _make_agent()
    result = agent.run(
        client_id="c1",
        request="classify",
        data={"columns": ["no_text_col", "label"], "rows": [{"no_text_col": "x", "label": 1}]},
        task="nlp",
        target_col="label",
    )
    assert result.success is False
    assert result.error is not None
