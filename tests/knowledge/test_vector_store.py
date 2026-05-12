import pytest
from src.knowledge.vector_store import VectorStore


@pytest.fixture
def store(tmp_path, mocker):
    mocker.patch(
        "chromadb.utils.embedding_functions.OpenAIEmbeddingFunction.__call__",
        return_value=[[0.1] * 128],
    )
    return VectorStore(persist_dir=str(tmp_path), openai_api_key="fake-key")


def test_add_and_search(store):
    store.add(
        client_id="client-1",
        texts=["NPL means non-performing loans overdue by 90+ days"],
        ids=["doc-1"],
    )
    results = store.search("client-1", "what is NPL", top_k=1)
    assert len(results) == 1
    assert "NPL" in results[0]


def test_client_isolation(store):
    store.add("client-a", ["Client A secret data"], ["id-a"])
    store.add("client-b", ["Client B secret data"], ["id-b"])

    results_a = store.search("client-a", "secret data", top_k=5)
    assert all("Client A" in r for r in results_a)

    results_b = store.search("client-b", "secret data", top_k=5)
    assert all("Client B" in r for r in results_b)


def test_delete_client(store):
    store.add("client-del", ["some data"], ["id-1"])
    store.delete_client("client-del")
    results = store.search("client-del", "some data", top_k=1)
    assert results == []
