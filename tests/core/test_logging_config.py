# tests/core/test_logging_config.py
import logging

from src.core.logging_config import _RequestIdFilter
from src.core.request_context import set_request_id


def test_filter_injects_request_id():
    set_request_id("test-req-id")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    f = _RequestIdFilter()
    f.filter(record)
    assert record.request_id == "test-req-id"


def test_filter_injects_empty_string_when_no_request():
    set_request_id("")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    f = _RequestIdFilter()
    f.filter(record)
    assert record.request_id == ""
