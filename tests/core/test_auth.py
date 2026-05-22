# tests/core/test_auth.py
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from src.core.auth import make_verify_api_key, make_verify_client_api_key


def _settings(api_key=""):
    s = MagicMock()
    s.api_key = api_key
    return s


def test_correct_key_does_not_raise():
    verify = make_verify_api_key(_settings(api_key="secret"))
    verify(x_api_key="secret")  # must not raise


def test_wrong_key_raises_401():
    verify = make_verify_api_key(_settings(api_key="secret"))
    with pytest.raises(HTTPException) as exc_info:
        verify(x_api_key="wrong")
    assert exc_info.value.status_code == 401
    assert "Invalid" in exc_info.value.detail


def test_missing_key_raises_401():
    verify = make_verify_api_key(_settings(api_key="secret"))
    with pytest.raises(HTTPException) as exc_info:
        verify(x_api_key="")
    assert exc_info.value.status_code == 401


def test_empty_settings_key_disables_auth():
    verify = make_verify_api_key(_settings(api_key=""))
    verify(x_api_key="")          # must not raise
    verify(x_api_key="anything")  # must not raise


def _make_registry(client=None):
    r = MagicMock()
    r.get.return_value = client
    return r


def _make_client_config(api_key="secret"):
    c = MagicMock()
    c.api_key = api_key
    return c


def test_verify_client_api_key_correct_key():
    registry = _make_registry(client=_make_client_config(api_key="secret"))
    verify = make_verify_client_api_key(registry)
    verify(x_client_id="c1", x_api_key="secret")  # must not raise


def test_verify_client_api_key_wrong_key_raises_401():
    registry = _make_registry(client=_make_client_config(api_key="secret"))
    verify = make_verify_client_api_key(registry)
    with pytest.raises(HTTPException) as exc_info:
        verify(x_client_id="c1", x_api_key="wrong")
    assert exc_info.value.status_code == 401
    assert "Invalid" in exc_info.value.detail


def test_verify_client_api_key_missing_client_raises_401():
    registry = _make_registry(client=None)
    verify = make_verify_client_api_key(registry)
    with pytest.raises(HTTPException) as exc_info:
        verify(x_client_id="unknown", x_api_key="anything")
    assert exc_info.value.status_code == 401
