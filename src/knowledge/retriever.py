from src.knowledge.vector_store import VectorStore


class KnowledgeRetriever:
    def __init__(self, store: VectorStore, top_k: int = 5):
        self._store = store
        self._top_k = top_k

    def search(self, client_id: str, query: str) -> list[str]:
        return self._store.search(client_id, query, self._top_k)
