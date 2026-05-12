from unittest.mock import MagicMock
from src.knowledge.retriever import KnowledgeRetriever


def test_search_delegates_to_store():
    store = MagicMock()
    store.search.return_value = ["NPL means 90+ days overdue", "Credit risk defined as..."]
    retriever = KnowledgeRetriever(store=store, top_k=5)

    results = retriever.search("client-1", "what is NPL")

    store.search.assert_called_once_with("client-1", "what is NPL", 5)
    assert len(results) == 2


def test_search_returns_empty_when_no_knowledge():
    store = MagicMock()
    store.search.return_value = []
    retriever = KnowledgeRetriever(store=store)

    results = retriever.search("client-new", "anything")
    assert results == []
