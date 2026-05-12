from unittest.mock import MagicMock, patch
from src.core.llm import LLMRouter
from src.core.models import TaskType


def make_router():
    return LLMRouter(anthropic_api_key="fake-anthropic", openai_api_key="fake-openai")


def test_reasoning_routes_to_claude(mocker):
    router = make_router()
    mock_complete = mocker.patch.object(router, "_claude", return_value="claude reply")
    result = router.complete(TaskType.REASONING, "sys", "user")
    mock_complete.assert_called_once_with("sys", "user", model="claude-sonnet-4-6")
    assert result == "claude reply"


def test_tool_routes_to_gpt(mocker):
    router = make_router()
    mock_complete = mocker.patch.object(router, "_gpt", return_value="gpt reply")
    result = router.complete(TaskType.TOOL, "sys", "user")
    mock_complete.assert_called_once_with("sys", "user", model="gpt-4o")
    assert result == "gpt reply"


def test_simple_routes_to_haiku(mocker):
    router = make_router()
    mock_complete = mocker.patch.object(router, "_claude", return_value="haiku reply")
    result = router.complete(TaskType.SIMPLE, "sys", "user")
    mock_complete.assert_called_once_with("sys", "user", model="claude-haiku-4-5-20251001")
    assert result == "haiku reply"
