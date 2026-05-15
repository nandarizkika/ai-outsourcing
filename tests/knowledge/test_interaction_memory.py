import json
import pytest
from unittest.mock import MagicMock, call

from src.knowledge.interaction_memory import InteractionMemoryLogger
from src.core.models import Channel, Request, Response


def _make_request(text="show churn", client_id="c1"):
    return Request(
        channel=Channel.SLACK,
        sender_id="U1",
        sender_name="Ana",
        text=text,
        timestamp="2026-05-14T00:00:00",
        client_id=client_id,
    )


def test_log_calls_store_add():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="Churn is 5%.")
    logger.log(req, resp)
    mock_store.add.assert_called_once()


def test_log_uses_correct_client_id():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request(client_id="client-xyz")
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp)
    args = mock_store.add.call_args[0]
    assert args[0] == "client-xyz"


def test_log_document_contains_request_and_response():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request(text="show revenue")
    resp = Response(request_id="r1", text="Revenue is $1M.")
    logger.log(req, resp)
    doc_str = mock_store.add.call_args[0][1][0]
    doc = json.loads(doc_str)
    assert doc["request_text"] == "show revenue"
    assert doc["response_text"] == "Revenue is $1M."
    assert doc["channel"] == "slack"


def test_log_includes_sql_queries():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp, sql_queries=["SELECT churn FROM metrics"])
    doc = json.loads(mock_store.add.call_args[0][1][0])
    assert doc["sql_queries"] == ["SELECT churn FROM metrics"]


def test_log_metadata_has_type_interaction_memory():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp)
    metadatas = mock_store.add.call_args[0][3]
    assert metadatas[0]["type"] == "interaction_memory"


def test_log_generates_unique_ids():
    mock_store = MagicMock()
    logger = InteractionMemoryLogger(store=mock_store)
    req = _make_request()
    resp = Response(request_id="r1", text="ok")
    logger.log(req, resp)
    logger.log(req, resp)
    ids_first = mock_store.add.call_args_list[0][0][2]
    ids_second = mock_store.add.call_args_list[1][0][2]
    assert ids_first[0] != ids_second[0]
