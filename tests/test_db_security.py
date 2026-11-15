"""The `agent_reader` role can read the data tables and do nothing else."""

import psycopg.errors
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.usefixtures("db")

DATA_TABLES = [
    "companies",
    "income_statements",
    "balance_sheets",
    "cash_flows",
    "daily_prices",
    "dividends",
    "stock_splits",
    "income_statements_split_adjusted",
]

FORBIDDEN = [
    "INSERT INTO companies (ticker, name) VALUES ('ZZZZ', 'Nope Inc')",
    "UPDATE companies SET name = 'x'",
    "DELETE FROM companies",
    "INSERT INTO stock_splits (ticker, split_date, ratio) VALUES ('AAPL', '2020-01-01', 2)",
    "DROP VIEW income_statements_split_adjusted",
    "TRUNCATE companies CASCADE",
    "DROP TABLE companies CASCADE",
    "ALTER TABLE companies ADD COLUMN evil TEXT",
    "CREATE TABLE evil (id INT)",
    "CREATE TEMP TABLE evil (id INT)",
    "CREATE SCHEMA evil",
    "INSERT INTO schema_docs (table_name, description) VALUES ('t', 'd')",
    "INSERT INTO query_examples (question, sql) VALUES ('q', 'SELECT 1')",
    "DELETE FROM schema_migrations",
    "SELECT * FROM schema_migrations",
    "COPY companies TO PROGRAM 'cat'",
    "SELECT pg_read_file('/etc/passwd')",
]


def _run_as_reader(statement: str) -> None:
    from app.db.session import reader_engine

    with reader_engine.connect() as conn:
        conn.execute(text(statement))
        conn.commit()


@pytest.mark.parametrize("table", DATA_TABLES)
def test_reader_can_select(table: str) -> None:
    _run_as_reader(f"SELECT * FROM {table} LIMIT 1")


@pytest.mark.parametrize("statement", FORBIDDEN)
def test_reader_is_denied(statement: str) -> None:
    with pytest.raises(DBAPIError) as excinfo:
        _run_as_reader(statement)

    assert isinstance(excinfo.value.orig, psycopg.errors.InsufficientPrivilege)


def test_reader_is_who_we_think() -> None:
    from app.db.session import reader_engine

    with reader_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT current_user, rolsuper, rolcreatedb, rolcreaterole "
                "FROM pg_roles WHERE rolname = current_user"
            )
        ).one()

    assert tuple(row) == ("agent_reader", False, False, False)


def test_reader_statement_timeout_is_set() -> None:
    from app.db.session import reader_engine

    with reader_engine.connect() as conn:
        assert conn.execute(text("SHOW statement_timeout")).scalar_one() == "5s"


def test_role_default_timeout_applies_without_engine_options() -> None:
    """A connection that skips `reader_engine` still gets the 5 s limit from the role."""
    from app.config import get_settings

    plain = create_engine(get_settings().reader_database_url.get_secret_value())
    try:
        with plain.connect() as conn:
            assert conn.execute(text("SHOW statement_timeout")).scalar_one() == "5s"
    finally:
        plain.dispose()


def test_reader_password_is_stored_hashed() -> None:
    from app.db.session import admin_engine

    with admin_engine.connect() as conn:
        stored = conn.execute(
            text("SELECT rolpassword FROM pg_authid WHERE rolname = 'agent_reader'")
        ).scalar_one()

    assert stored.startswith("SCRAM-SHA-256$")


def test_new_tables_are_not_readable_by_default() -> None:
    """A table added later gives the reader nothing until a migration grants it."""
    from app.db.session import admin_engine

    with admin_engine.connect() as conn:
        conn.execute(text("CREATE TABLE future_table (id INT)"))
        readable = conn.execute(
            text("SELECT has_table_privilege('agent_reader', 'future_table', 'SELECT')")
        ).scalar_one()
        conn.rollback()

    assert readable is False


def test_slow_query_is_cancelled() -> None:
    with pytest.raises(DBAPIError) as excinfo:
        _run_as_reader("SELECT pg_sleep(6)")

    assert isinstance(excinfo.value.orig, psycopg.errors.QueryCanceled)
