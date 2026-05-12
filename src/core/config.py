# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    slack_bot_token: str
    slack_signing_secret: str
    chromadb_persist_dir: str = ".chromadb"

    model_config = SettingsConfigDict(env_file=".env")
