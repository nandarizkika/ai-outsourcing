# tests/connectors/test_factory.py
import pytest
from unittest.mock import patch, MagicMock

from src.connectors.factory import ConnectorFactory
from src.connectors.sql import SQLConnector


def _create(connector_type, connector_config):
    with patch("src.connectors.factory.create_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        conn = ConnectorFactory.create(connector_type, connector_config)
    return conn, mock_engine


def test_postgres_builds_correct_url():
    conn, mock_engine = _create("postgres", {
        "host": "db.example.com", "database": "mydb", "user": "admin", "password": "secret"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "postgresql+psycopg2://admin:secret@db.example.com:5432/mydb"


def test_postgres_default_port():
    _, mock_engine = _create("postgres", {
        "host": "localhost", "database": "mydb", "user": "u", "password": "p"
    })
    assert ":5432/" in mock_engine.call_args[0][0]


def test_mysql_builds_correct_url():
    conn, mock_engine = _create("mysql", {
        "host": "db.example.com", "database": "mydb", "user": "admin", "password": "secret"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "mysql+pymysql://admin:secret@db.example.com:3306/mydb"


def test_mysql_default_port():
    _, mock_engine = _create("mysql", {
        "host": "localhost", "database": "mydb", "user": "u", "password": "p"
    })
    assert ":3306/" in mock_engine.call_args[0][0]


def test_bigquery_builds_correct_url():
    conn, mock_engine = _create("bigquery", {
        "project_id": "my-project", "dataset": "analytics"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "bigquery://my-project/analytics"


def test_snowflake_builds_correct_url():
    conn, mock_engine = _create("snowflake", {
        "account": "myaccount", "user": "admin", "password": "secret",
        "warehouse": "COMPUTE_WH", "database": "mydb"
    })
    assert isinstance(conn, SQLConnector)
    url = mock_engine.call_args[0][0]
    assert url == "snowflake://admin:secret@myaccount/mydb/PUBLIC?warehouse=COMPUTE_WH"


def test_snowflake_custom_schema():
    _, mock_engine = _create("snowflake", {
        "account": "myaccount", "user": "admin", "password": "secret",
        "warehouse": "COMPUTE_WH", "database": "mydb", "schema": "RAW"
    })
    assert "/RAW?" in mock_engine.call_args[0][0]


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown connector_type"):
        ConnectorFactory.create("oracle", {})
