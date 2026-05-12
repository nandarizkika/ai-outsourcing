from src.agents.chart_agent import ChartAgent


def make_agent():
    return ChartAgent()


def test_single_value_skips_chart():
    agent = make_agent()
    data = {"rows": [{"count": 42}], "columns": ["count"]}
    result = agent.run(data)
    assert result.success is True
    assert result.data == {"skipped": True}
    assert result.chart_png is None


def test_categorical_data_produces_png():
    agent = make_agent()
    data = {
        "rows": [
            {"region": "Jakarta", "total": 1000},
            {"region": "Surabaya", "total": 2000},
            {"region": "Bandung", "total": 500},
        ],
        "columns": ["region", "total"],
    }
    result = agent.run(data)
    assert result.success is True
    assert isinstance(result.chart_png, bytes)
    assert result.chart_png[:4] == b"\x89PNG"


def test_empty_data_returns_error():
    agent = make_agent()
    result = agent.run({"rows": [], "columns": ["x", "y"]})
    assert result.success is False
    assert result.error is not None


def test_missing_data_returns_error():
    agent = make_agent()
    result = agent.run({})
    assert result.success is False


def test_time_series_data_produces_png():
    agent = make_agent()
    data = {
        "rows": [{"month": i, "revenue": i * 100} for i in range(1, 7)],
        "columns": ["month", "revenue"],
    }
    result = agent.run(data)
    assert result.success is True
    assert isinstance(result.chart_png, bytes)
