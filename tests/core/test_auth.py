# tests/core/test_auth.py
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from src.core.auth import make_verify_api_key


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
