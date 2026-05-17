import json
import threading

from src.connectors.sql import SQLConnector
from src.core.models import ClientConfig


class ClientRegistry:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._lock = threading.Lock()
        self._connectors: dict[str, SQLConnector] = {}
        try:
            with open(filepath, "r") as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(filepath, "w") as f:
                json.dump({}, f)

    def _load(self) -> dict[str, dict]:
        with open(self._filepath, "r") as f:
            return json.load(f)

    def _save(self, data: dict[str, dict]) -> None:
        with open(self._filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, client_id: str) -> ClientConfig | None:
        with self._lock:
            raw = self._load().get(client_id)
            return ClientConfig(**raw) if raw is not None else None

    def upsert(self, config: ClientConfig) -> None:
        with self._lock:
            data = self._load()
            data[config.client_id] = config.model_dump()
            self._save(data)
            self._connectors.pop(config.client_id, None)

    def delete(self, client_id: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(client_id, None)
            self._save(data)
            self._connectors.pop(client_id, None)

    def all(self) -> list[ClientConfig]:
        with self._lock:
            return [ClientConfig(**v) for v in self._load().values()]

    def get_connector(self, client_id: str) -> SQLConnector | None:
        with self._lock:
            raw = self._load().get(client_id)
            if raw is None:
                return None
            db_url = raw.get("database_url")
            if not db_url:
                return None
            if client_id not in self._connectors:
                self._connectors[client_id] = SQLConnector(connection_url=db_url)
            return self._connectors[client_id]
