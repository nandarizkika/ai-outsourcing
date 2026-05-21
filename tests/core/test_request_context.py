# tests/core/test_request_context.py
from src.core.request_context import get_request_id, set_request_id


def test_get_request_id_default_is_empty_string():
    set_request_id("")
    assert get_request_id() == ""


def test_set_and_get_request_id():
    set_request_id("abc123")
    assert get_request_id() == "abc123"


def test_request_id_isolated_per_context():
    from contextvars import copy_context
    set_request_id("outer")
    inner_value = []

    def run_in_copy():
        set_request_id("inner")
        inner_value.append(get_request_id())

    copy_context().run(run_in_copy)
    assert inner_value[0] == "inner"
    assert get_request_id() == "outer"
