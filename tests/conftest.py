"""Shared fixtures."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError


@pytest.fixture(scope="session")
def db() -> None:
    """A migrated database. Tests that use it are skipped when Postgres is unreachable."""
    from app.config import get_settings
    from app.db.session import admin_engine
    from db.migrate import run

    try:
        with admin_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("Postgres is unreachable; start it with `docker compose up -d db`")
    run(admin_engine, get_settings().reader_database_url.get_secret_value())
