# tests/agents/test_segmentation_agent.py
import numpy as np
from unittest.mock import MagicMock

from src.agents.segmentation_agent import SegmentationAgent


def _make_agent():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Segment 0 is low-value, Segment 1 is high-value."
    return SegmentationAgent(llm=mock_llm)


def _clusterable_data(n=20):
    rng = np.random.default_rng(42)
    cluster1 = [{"revenue": float(r), "sessions": float(s)}
                for r, s in zip(rng.normal(100, 5, n), rng.normal(50, 3, n))]
    cluster2 = [{"revenue": float(r), "sessions": float(s)}
                for r, s in zip(rng.normal(500, 10, n), rng.normal(200, 8, n))]
    return {"columns": ["revenue", "sessions"], "rows": cluster1 + cluster2}


def test_segmentation_returns_n_segments():
    agent = _make_agent()
    result = agent.run("c1", "Segment customers", _clusterable_data())
    assert result.success is True
    assert "n_segments" in result.data
    assert result.data["n_segments"] >= 2


def test_segmentation_returns_chart_png():
    agent = _make_agent()
    result = agent.run("c1", "Segment customers", _clusterable_data())
    assert result.chart_png is not None
    assert len(result.chart_png) > 0


def test_segmentation_returns_centroids():
    agent = _make_agent()
    result = agent.run("c1", "Segment", _clusterable_data())
    assert "centroids" in result.data
    assert len(result.data["centroids"]) == result.data["n_segments"]


def test_rows_have_segment_column():
    agent = _make_agent()
    result = agent.run("c1", "Segment", _clusterable_data())
    assert "rows" in result.data
    assert all("segment" in r for r in result.data["rows"])


def test_silhouette_score_in_data():
    agent = _make_agent()
    result = agent.run("c1", "Segment", _clusterable_data())
    assert "silhouette_score" in result.data
    assert isinstance(result.data["silhouette_score"], float)


def test_insufficient_columns_returns_error():
    agent = _make_agent()
    data = {"columns": ["revenue"], "rows": [{"revenue": float(i)} for i in range(10)]}
    result = agent.run("c1", "Segment", data)
    assert result.success is False
    assert result.error is not None


def test_too_few_rows_returns_error():
    agent = _make_agent()
    data = {"columns": ["x", "y"], "rows": [{"x": 1.0, "y": 2.0}]}
    result = agent.run("c1", "Segment", data)
    assert result.success is False
