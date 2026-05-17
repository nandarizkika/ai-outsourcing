# tests/core/test_models_phase7.py
from src.core.models import ClientConfig, Response, Tier, Channel
from src.core.config import Settings


def _base_config(**kwargs):
    return dict(
        client_id="c1", name="Test", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor",
        active_channels=[Channel.SLACK],
        **kwargs,
    )


def test_client_config_database_url_defaults_to_none():
    config = ClientConfig(**_base_config())
    assert config.database_url is None


def test_client_config_accepts_database_url():
    config = ClientConfig(**_base_config(database_url="postgresql://localhost/mydb"))
    assert config.database_url == "postgresql://localhost/mydb"


def test_response_report_markdown_defaults_to_none():
    r = Response(request_id="r1", text="hello")
    assert r.report_markdown is None


def test_response_report_html_defaults_to_none():
    r = Response(request_id="r1", text="hello")
    assert r.report_html is None


def test_settings_clients_file_default():
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "x")
    os.environ.setdefault("OPENAI_API_KEY", "x")
    os.environ.setdefault("SLACK_BOT_TOKEN", "x")
    os.environ.setdefault("SLACK_SIGNING_SECRET", "x")
    s = Settings()
    assert s.clients_file == "clients.json"
