# tests/core/test_config.py
import os
import pytest
from src.core.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("CHROMADB_PERSIST_DIR", ".test-chroma")

    s = Settings()
    assert s.anthropic_api_key == "test-anthropic"
    assert s.openai_api_key == "test-openai"
    assert s.chromadb_persist_dir == ".test-chroma"


def test_settings_has_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "c")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "d")

    s = Settings()
    assert s.chromadb_persist_dir == ".chromadb"


def test_jira_settings_default_empty():
    from src.core.config import Settings
    s = Settings(
        anthropic_api_key="a",
        openai_api_key="b",
        slack_bot_token="c",
        slack_signing_secret="d",
    )
    assert s.jira_url == ""
    assert s.jira_email == ""
    assert s.jira_api_token == ""
    assert s.jira_project_key == "AI"


def test_email_settings_default_empty():
    from src.core.config import Settings
    s = Settings(
        anthropic_api_key="a",
        openai_api_key="b",
        slack_bot_token="c",
        slack_signing_secret="d",
    )
    assert s.email_imap_host == ""
    assert s.email_imap_user == ""
    assert s.email_smtp_port == 587


def test_slack_bot_user_id_defaults_to_empty_string():
    s = Settings(
        anthropic_api_key="a",
        openai_api_key="b",
        slack_bot_token="c",
        slack_signing_secret="d",
    )
    assert s.slack_bot_user_id == ""


def test_slack_bot_user_id_loaded_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "c")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "d")
    monkeypatch.setenv("SLACK_BOT_USER_ID", "U0123456789")
    s = Settings()
    assert s.slack_bot_user_id == "U0123456789"


def test_log_level_defaults_to_info():
    s = Settings(
        anthropic_api_key="a",
        openai_api_key="b",
        slack_bot_token="c",
        slack_signing_secret="d",
    )
    assert s.log_level == "INFO"
