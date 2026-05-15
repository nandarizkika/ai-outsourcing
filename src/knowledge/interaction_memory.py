import json
import uuid

from src.core.models import Request, Response
from src.knowledge.vector_store import VectorStore


class InteractionMemoryLogger:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def log(
        self,
        request: Request,
        response: Response,
        sql_queries: list[str] | None = None,
        corrections: list[str] | None = None,
    ) -> None:
        doc = json.dumps({
            "request_text": request.text,
            "response_text": response.text,
            "sql_queries": sql_queries or [],
            "corrections": corrections or [],
            "timestamp": request.timestamp,
            "channel": request.channel.value,
            "client_id": request.client_id,
        })
        self._store.add(
            request.client_id,
            [doc],
            [str(uuid.uuid4())],
            [{"type": "interaction_memory", "client_id": request.client_id}],
        )
