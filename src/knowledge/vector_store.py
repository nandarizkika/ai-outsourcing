import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from chromadb.utils import embedding_functions


class _OpenAIEmbeddingWrapper(EmbeddingFunction[Documents]):
    """Thin wrapper around OpenAIEmbeddingFunction with an explicit __call__
    signature so ChromaDB's signature validator accepts it in tests where the
    inner function is mocked."""

    def __init__(self, openai_ef: embedding_functions.OpenAIEmbeddingFunction):
        self._inner = openai_ef

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        return self._inner(input)


class VectorStore:
    def __init__(self, persist_dir: str, openai_api_key: str = ""):
        self._client = chromadb.PersistentClient(path=persist_dir)

        if openai_api_key:
            _inner_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            )
            self._ef = _OpenAIEmbeddingWrapper(_inner_ef)
        else:
            self._ef = embedding_functions.DefaultEmbeddingFunction()

    def _collection(self, client_id: str):
        return self._client.get_or_create_collection(
            name=f"client_{client_id}",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        client_id: str,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        col = self._collection(client_id)
        col.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas if metadatas else None,
        )

    def search(self, client_id: str, query: str, top_k: int = 5) -> list[str]:
        try:
            col = self._collection(client_id)
            if col.count() == 0:
                return []
            results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    def delete_client(self, client_id: str) -> None:
        try:
            self._client.delete_collection(f"client_{client_id}")
        except Exception:
            pass
