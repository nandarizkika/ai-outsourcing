# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str
    chromadb_persist_dir: str = ".chromadb"
    clients_file: str = "clients.json"
    api_key: str = ""
    log_level: str = "INFO"
    slack_bot_user_id: str = ""

    # Jira settings
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "AI"

    # Email settings
    email_imap_host: str = ""
    email_imap_user: str = ""
    email_imap_password: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587

    model_config = SettingsConfigDict(env_file=".env")
