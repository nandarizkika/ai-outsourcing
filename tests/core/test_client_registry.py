# tests/core/test_client_registry.py
import pytest
from unittest.mock import patch

from src.core.client_registry import ClientRegistry
from src.core.models import ClientConfig, Tier, Channel


def _make_config(client_id="c1", database_url=None, connector_type=None, connector_config=None):
    return ClientConfig(
        client_id=client_id,
        name=f"Client {client_id}",
        tier=Tier.BASIC,
        enabled_skills=[],
        account_mode="vendor",
        active_channels=[Channel.SLACK],
        database_url=database_url,
        connector_type=connector_type,
        connector_config=connector_config,
    )


def test_upsert_and_get_returns_config(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    config = _make_config("c1")
    reg.upsert(config)
    result = reg.get("c1")
    assert result is not None
    assert result.client_id == "c1"
    assert result.name == "Client c1"


def test_get_missing_returns_none(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    assert reg.get("nonexistent") is None


def test_delete_removes_config(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    reg.upsert(_make_config("c1"))
    reg.delete("c1")
    assert reg.get("c1") is None


def test_all_returns_all_configs(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    reg.upsert(_make_config("c1"))
    reg.upsert(_make_config("c2"))
    all_configs = reg.all()
    assert len(all_configs) == 2
    ids = {c.client_id for c in all_configs}
    assert ids == {"c1", "c2"}


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "clients.json")
    reg1 = ClientRegistry(path)
    reg1.upsert(_make_config("c1"))
    reg2 = ClientRegistry(path)
    result = reg2.get("c1")
    assert result is not None
    assert result.client_id == "c1"


def test_get_connector_returns_none_without_database_url(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    reg.upsert(_make_config("c1", database_url=None))
    assert reg.get_connector("c1") is None


def test_get_connector_caches_instance(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    reg.upsert(_make_config("c1", database_url="sqlite:///:memory:"))
    conn1 = reg.get_connector("c1")
    conn2 = reg.get_connector("c1")
    assert conn1 is conn2


def test_upsert_with_new_url_evicts_connector(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    reg.upsert(_make_config("c1", database_url="sqlite:///a.db"))
    conn1 = reg.get_connector("c1")
    reg.upsert(_make_config("c1", database_url="sqlite:///b.db"))
    conn2 = reg.get_connector("c1")
    assert conn1 is not conn2


def test_delete_evicts_connector_cache(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    reg.upsert(_make_config("c1", database_url="sqlite:///:memory:"))
    conn1 = reg.get_connector("c1")
    reg.delete("c1")
    reg.upsert(_make_config("c1", database_url="sqlite:///:memory:"))
    conn2 = reg.get_connector("c1")
    assert conn1 is not conn2


def test_client_config_accepts_connector_type_and_config(tmp_path):
    reg = ClientRegistry(str(tmp_path / "clients.json"))
    config = ClientConfig(
        client_id="c1", name="Client c1", tier=Tier.BASIC,
        enabled_skills=[], account_mode="vendor", active_channels=[Channel.SLACK],
        connector_type="postgres",
        connector_config={"host": "localhost", "database": "mydb", "user": "u", "password": "p"},
    )
    reg.upsert(config)
    result = reg.get("c1")
    assert result.connector_type == "postgres"
    assert result.connector_config["host"] == "localhost"
