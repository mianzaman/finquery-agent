"""Shared fixtures."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach a real model: asking for the configured one fails the test.

    Tests pass a fake model explicitly, or replace the function that would call one.
    """
    from app import llm

    def refuse() -> None:
        raise AssertionError(
            "a test tried to use the real chat model; pass a fake model or patch the caller"
        )

    monkeypatch.setattr(llm, "get_chat_model", refuse)


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
