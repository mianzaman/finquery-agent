"""Database engines: admin_engine and reader_engine."""

from sqlalchemy import Engine, create_engine

from app.config import get_settings

READER_STATEMENT_TIMEOUT_MS = 5000


def _make_admin_engine() -> Engine:
    """Engine for migrations and ingestion. Never used by agent code."""
    url = get_settings().database_url.get_secret_value()
    return create_engine(url, pool_pre_ping=True)


def _make_reader_engine() -> Engine:
    """Engine for the `agent_reader` role: SELECT only, statements cancelled after 5 s."""
    url = get_settings().reader_database_url.get_secret_value()
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"options": f"-c statement_timeout={READER_STATEMENT_TIMEOUT_MS}"},
    )


admin_engine: Engine = _make_admin_engine()
reader_engine: Engine = _make_reader_engine()
