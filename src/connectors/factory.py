# src/connectors/factory.py
from urllib.parse import quote_plus

from sqlalchemy import create_engine

from src.connectors.sql import SQLConnector


def _postgres_url(cfg: dict) -> str:
    return (
        f"postgresql+psycopg2://{quote_plus(cfg['user'])}:{quote_plus(cfg['password'])}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )


def _mysql_url(cfg: dict) -> str:
    return (
        f"mysql+pymysql://{quote_plus(cfg['user'])}:{quote_plus(cfg['password'])}"
        f"@{cfg['host']}:{cfg.get('port', 3306)}/{cfg['database']}"
    )


def _bigquery_url(cfg: dict) -> str:
    return f"bigquery://{cfg['project_id']}/{cfg['dataset']}"


def _snowflake_url(cfg: dict) -> str:
    schema = cfg.get("schema", "PUBLIC")
    return (
        f"snowflake://{quote_plus(cfg['user'])}:{quote_plus(cfg['password'])}"
        f"@{cfg['account']}/{cfg['database']}/{schema}"
        f"?warehouse={cfg['warehouse']}"
    )


_BUILDERS = {
    "postgres": _postgres_url,
    "mysql": _mysql_url,
    "bigquery": _bigquery_url,
    "snowflake": _snowflake_url,
}


class ConnectorFactory:
    @staticmethod
    def create(connector_type: str, connector_config: dict) -> SQLConnector:
        builder = _BUILDERS.get(connector_type)
        if builder is None:
            raise ValueError(f"Unknown connector_type: {connector_type!r}")
        url = builder(connector_config)
        engine = create_engine(url)
        return SQLConnector(connection_url=url, engine=engine)
