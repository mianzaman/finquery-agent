"""Tests for `run_select`: what the database enforces even if the SQL checks miss something."""

import pytest

from app.db.session import QueryError, run_select

pytestmark = pytest.mark.usefixtures("db")


def test_returns_columns_rows_and_whether_more_were_available() -> None:
    sql = "SELECT n AS number, n * 2 AS doubled FROM generate_series(1, 5) AS n ORDER BY n"

    assert run_select(sql) == (
        ["number", "doubled"],
        [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)],
        False,
    )
    assert run_select(sql, max_rows=5)[2] is False
    columns, rows, truncated = run_select(sql, max_rows=2)
    assert (columns, rows, truncated) == (["number", "doubled"], [(1, 2), (2, 4)], True)


def test_a_percent_sign_is_not_a_placeholder() -> None:
    assert run_select("SELECT 'A%' AS pattern, 7 % 4 AS remainder")[1] == [("A%", 3)]


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT 1; SELECT 2", "multiple commands"),
        ("SELECT 1; SET statement_timeout = 0", "multiple commands"),
        ("INSERT INTO companies (ticker, name) VALUES ('ZZZZ', 'x')", "read-only transaction"),
        ("CREATE TEMP TABLE evil (id INT)", "read-only transaction"),
        ("SELECT nope FROM companies", "does not exist"),
        ("SET statement_timeout = 0", "no result set"),
        ("SELECT * FROM schema_migrations", "permission denied"),
    ],
)
def test_the_database_refuses(sql: str, message: str) -> None:
    with pytest.raises(QueryError, match=message):
        run_select(sql)


def test_a_slow_query_is_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.db.session.READER_STATEMENT_TIMEOUT_MS", 200)

    with pytest.raises(QueryError, match="statement timeout"):
        run_select("SELECT pg_sleep(5)")


def test_the_timeout_and_read_only_mode_are_set_for_every_query() -> None:
    # A session-level SET is rolled back with its transaction and the next query sets its own.
    with pytest.raises(QueryError):
        run_select("SET statement_timeout = 0")

    assert run_select("SHOW statement_timeout")[1] == [("5s",)]
    assert run_select("SHOW transaction_read_only")[1] == [("on",)]


def test_an_unreachable_database_is_a_query_error_without_connection_details() -> None:
    from sqlalchemy import create_engine

    nowhere = create_engine("postgresql+psycopg://nobody:secret_pw@127.0.0.1:1/none")

    with pytest.raises(QueryError) as excinfo:
        run_select("SELECT 1", engine=nowhere)

    assert str(excinfo.value) == "the database is unreachable"
    assert "secret_pw" not in str(excinfo.value) and "127.0.0.1" not in str(excinfo.value)


def test_the_error_message_is_one_line_without_connection_details() -> None:
    with pytest.raises(QueryError) as excinfo:
        run_select("SELECT nope FROM companies")

    assert "\n" not in str(excinfo.value)
    assert "agent_reader" not in str(excinfo.value) and "localhost" not in str(excinfo.value)
