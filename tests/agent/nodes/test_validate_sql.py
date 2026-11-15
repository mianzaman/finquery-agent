"""Tests for SQL validation: the pure checks, the row limit, and the node with EXPLAIN."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.agent.nodes import validate_sql as validate_module
from app.agent.nodes.validate_sql import (
    ALLOWED_TABLES,
    MAX_LIMIT,
    SqlRejected,
    check_sql,
    limit_rows,
    validate_sql,
)

ALLOWED = [
    "SELECT 1",
    "SELECT ticker, revenue FROM income_statements WHERE fiscal_year = 2025;",
    "select count(*), sum(revenue), max(period_end) from income_statements group by ticker",
    "WITH latest AS (SELECT ticker, max(trade_date) AS d FROM daily_prices GROUP BY ticker) "
    "SELECT p.ticker, p.close FROM daily_prices p JOIN latest l ON l.ticker = p.ticker "
    "AND l.d = p.trade_date ORDER BY p.close DESC LIMIT 5",
    "SELECT ticker FROM companies WHERE sector = 'Energy' UNION SELECT ticker FROM dividends",
    "(SELECT 1) UNION ALL (SELECT 2)",
    "SELECT name FROM companies WHERE name LIKE 'A%' -- a comment; with a semicolon",
    "SELECT 'DROP TABLE companies; DELETE' AS harmless_text",
    "SELECT date_trunc('year', trade_date), avg(close), now() FROM daily_prices GROUP BY 1",
    "SELECT ticker, RANK() OVER (ORDER BY revenue DESC) FROM income_statements",
    "SELECT * FROM income_statements_split_adjusted WHERE ticker = 'NVDA'",
    "SELECT * FROM public.companies",
    "SELECT c.ticker, s.ratio FROM companies c LEFT JOIN stock_splits s ON s.ticker = c.ticker",
    "SELECT ticker, eps_diluted_adjusted FROM income_statements_split_adjusted",
    "SELECT ticker FROM balance_sheets WHERE ticker IN (SELECT ticker FROM cash_flows)",
    # A CTE may have any name: inside its scope the name means the CTE, not a server table.
    "WITH pg_user AS (SELECT ticker FROM companies) SELECT * FROM pg_user",
    "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    "SELECT COALESCE(SUM(revenue), 0), NULLIF(MAX(revenue), 0), ROUND(AVG(revenue), 2), "
    "EXTRACT(YEAR FROM period_end), LAG(revenue) OVER (PARTITION BY ticker ORDER BY period_end), "
    "CASE WHEN revenue > 0 THEN 'up' ELSE 'down' END, revenue::numeric, "
    "TO_CHAR(period_end, 'YYYY-MM'), STRING_AGG(ticker, ','), COUNT(DISTINCT ticker), "
    "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue) FROM income_statements GROUP BY "
    "ticker, revenue, period_end",
    "SELECT ticker FROM companies WHERE ticker = ANY(ARRAY['AAPL', 'MSFT'])",
    "SELECT u FROM unnest(ARRAY[1, 2, 3]) AS u",
    'SELECT c."user" FROM (SELECT ticker AS "user" FROM companies) AS c',  # a quoted alias
]

REJECTED = [
    # (sql, a word the reason must contain)
    ("", "no SQL"),
    ("-- only a comment", "no SQL"),
    ("SELECT 1; SELECT 2", "2 statements"),
    ("SELECT 1; DROP TABLE companies", "2 statements"),
    ("SELECT FROM WHERE", "parsed"),
    ("INSERT INTO companies (ticker, name) VALUES ('X', 'x')", "INSERT"),
    ("UPDATE companies SET name = 'x'", "UPDATE"),
    ("DELETE FROM companies", "DELETE"),
    ("DROP TABLE companies", "DROP"),
    ("TRUNCATE companies", "TRUNCATE"),
    ("CREATE TABLE evil (id INT)", "CREATE"),
    ("ALTER TABLE companies ADD COLUMN evil TEXT", "ALTER"),
    ("GRANT ALL ON companies TO agent_reader", "GRANT"),
    ("COPY companies TO STDOUT", "COPY"),
    ("COPY companies TO PROGRAM 'cat'", "COPY"),
    ("SET statement_timeout = 0", "SET"),
    ("SET LOCAL statement_timeout = 0", "SET"),
    ("RESET ALL", "COMMAND"),
    ("RESET statement_timeout", "COMMAND"),
    ("SHOW statement_timeout", "COMMAND"),
    ("EXPLAIN ANALYZE SELECT 1", "COMMAND"),
    ("CALL do_something()", "COMMAND"),
    ("DO $$ BEGIN PERFORM pg_sleep(10); END $$", "COMMAND"),
    ("PREPARE p AS SELECT 1", "COMMAND"),
    ("LOCK TABLE companies", "COMMAND"),
    ("BEGIN", "only SELECT"),
    ("COMMIT", "only SELECT"),
    ("VALUES (1)", "only SELECT"),
    # Writes and locks hidden inside a query.
    ("WITH gone AS (DELETE FROM companies RETURNING *) SELECT * FROM gone", "DELETE"),
    (
        "WITH new AS (INSERT INTO companies (ticker, name) VALUES ('X','x') RETURNING *) "
        "SELECT * FROM new",
        "INSERT",
    ),
    ("WITH u AS (UPDATE companies SET name = 'x' RETURNING *) SELECT * FROM u", "UPDATE"),
    ("SELECT * INTO copy_of FROM companies", "INTO"),
    ("SELECT * FROM companies FOR UPDATE", "LOCK"),
    ("SELECT * FROM companies FOR SHARE", "LOCK"),
    # Functions that get around the role or the timeout.
    ("SELECT pg_sleep(60)", "pg_sleep"),
    ("SELECT PG_SLEEP(60)", "pg_sleep"),
    ("SELECT pg_catalog.pg_sleep(60)", "pg_sleep"),
    ('SELECT "pg_sleep"(60)', "pg_sleep"),
    ("SELECT pg_sleep_for('5 minutes')", "pg_sleep_for"),
    ("SELECT * FROM pg_sleep(60)", "pg_sleep"),
    ("SELECT ticker FROM companies WHERE pg_sleep(60) IS NULL", "pg_sleep"),
    ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
    ("SELECT pg_catalog.set_config('statement_timeout', '0', false)", "set_config"),
    ("SELECT pg_terminate_backend(pid) FROM pg_stat_activity", "pg_terminate_backend"),
    ("SELECT pg_cancel_backend(1)", "pg_cancel_backend"),
    ("SELECT pg_advisory_lock(1)", "pg_advisory_lock"),
    ("SELECT pg_try_advisory_xact_lock(1)", "pg_try_advisory_xact_lock"),
    ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
    ("SELECT pg_ls_dir('.')", "pg_ls_dir"),
    ("SELECT lo_import('/etc/passwd')", "lo_import"),
    ("SELECT * FROM dblink('host=x', 'SELECT 1') AS t(a int)", "dblink"),
    ("SELECT query_to_xml('SELECT pg_sleep(60)', true, true, '')", "query_to_xml"),
    ("SELECT nextval('some_seq')", "nextval"),
    # Functions that are simply not on the allow-list.
    ("SELECT current_setting('statement_timeout')", "current_setting"),
    ("SELECT version()", "current_version"),
    ("SELECT current_user", "current_user"),
    ("SELECT inet_server_addr()", "inet_server_addr"),
    ("SELECT pg_relation_filepath('companies')", "pg_relation_filepath"),
    ("SELECT regexp_replace(name, '(a+)+$', 'x') FROM companies", "regexp_replace"),
    ("SELECT some_new_function(ticker) FROM companies", "some_new_function"),
    # Allowed once, removed after the review: they build huge values or endless rows.
    ("SELECT replace(name, 'a', 'aaaaaaaaaa') FROM companies", "replace"),
    ("SELECT g FROM generate_series(1, 4) AS g", "generate_series"),
    ("SELECT string_agg('x', ',') FROM generate_series(1, 30000000)", "generate_series"),
    # The session's role and database, and catalog lookups by cast.
    ("SELECT user", "user is not allowed"),
    ("SELECT current_role", "current_role is not allowed"),
    ("SELECT system_user", "system_user is not allowed"),
    ("SELECT session_user", "not allowed"),
    ("SELECT ticker FROM companies WHERE name = user", "user is not allowed"),
    ("SELECT 'agent_reader'::regrole::text", "a cast to regrole is not allowed"),
    ("SELECT CAST('pg_authid' AS regclass)", "a cast to regclass is not allowed"),
    ("SELECT 'pg_sleep'::regproc", "a cast to regproc is not allowed"),
    # Table names the parser cannot reduce to schema.table.
    ("SELECT * FROM a.b.c.d", "could not be understood"),
    # Tables that are not the data tables.
    ("SELECT * FROM pg_user", "table pg_user is not available"),
    ("SELECT * FROM pg_settings", "table pg_settings is not available"),
    ("SELECT * FROM pg_catalog.pg_class", "table pg_catalog.pg_class is not available"),
    ("SELECT * FROM information_schema.tables", "information_schema.tables is not available"),
    ("SELECT * FROM schema_docs", "table schema_docs is not available"),
    ("SELECT question, sql FROM query_examples", "table query_examples is not available"),
    ("SELECT * FROM schema_migrations", "table schema_migrations is not available"),
    ("SELECT * FROM other_schema.companies", "other_schema.companies is not available"),
    ("SELECT * FROM otherdb.public.companies", "otherdb.public.companies is not available"),
    ("SELECT ticker FROM companies WHERE ticker IN (SELECT usename FROM pg_user)", "pg_user"),
    ("SELECT c.ticker FROM companies c, LATERAL (SELECT * FROM pg_settings) s", "pg_settings"),
    ("(SELECT ticker FROM companies) UNION (SELECT rolname FROM pg_roles)", "pg_roles"),
    ("WITH x AS (SELECT * FROM pg_stat_activity) SELECT * FROM x", "pg_stat_activity"),
    # A CTE named like a server table hides it only inside its own scope.
    (
        "SELECT * FROM pg_user, (WITH pg_user AS (SELECT 1 AS a) SELECT * FROM pg_user) s",
        "table pg_user is not available",
    ),
    ("WITH x AS (SELECT 1) SELECT * FROM pg_catalog.x", "pg_catalog.x is not available"),
    # Functions that run a SQL string, so the string would escape every check above.
    ("SELECT * FROM ts_stat('SELECT set_config(''statement_timeout'', ''0'', false)')", "ts_stat"),
    ("SELECT word FROM pg_catalog.ts_stat('SELECT ticker::tsvector FROM companies')", "ts_stat"),
    ("SELECT c.ticker FROM companies c, LATERAL ts_stat('SELECT pg_sleep(9)') s", "ts_stat"),
    ("SELECT ts_rewrite('a'::tsquery, 'SELECT pg_sleep(9)')", "ts_rewrite"),
    ("SELECT query_to_xmlschema('SELECT 1', true, true, '')", "query_to_xmlschema"),
    ("SELECT cursor_to_xml('c', 10, true, true, '')", "cursor_to_xml"),
    ("SELECT table_to_xml('companies', true, true, '')", "table_to_xml"),
    ("SELECT schema_to_xml('public', true, true, '')", "schema_to_xml"),
    ("SELECT database_to_xmlschema(true, true, '')", "database_to_xmlschema"),
    ("SELECT * FROM crosstab('SELECT 1, 2, 3') AS t(a int, b int)", "crosstab"),
    # One call that allocates up to 1 GB on the server and outlasts the timeout.
    ("SELECT length(repeat('a', 900000000))", "repeat"),
    ("SELECT lpad('a', 900000000, 'x')", "pad"),
    ("SELECT rpad(name, 900000000) FROM companies", "pad"),
]


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        (123, "not text"),
        (b"SELECT 1", "not text"),
        (["SELECT 1"], "not text"),
        ("SELECT " + "(" * 5000 + "1" + ")" * 5000, "could not be parsed"),
        ("SELECT 1 /* " + "x" * 20_000 + " */", "longer than 20000 characters"),
        ("SELECT 'é\ud800'", "not valid text"),
    ],
    ids=["number", "bytes", "list", "deep nesting", "very long", "lone surrogate"],
)
def test_odd_input_is_rejected_instead_of_crashing(sql: object, reason: str) -> None:
    with pytest.raises(SqlRejected, match=reason):
        check_sql(sql)  # type: ignore[arg-type]


@pytest.mark.parametrize("sql", ALLOWED)
def test_read_only_queries_pass_unchanged(sql: str) -> None:
    assert check_sql(sql) == sql


@pytest.mark.parametrize(("sql", "reason"), REJECTED, ids=[sql[:40] for sql, _ in REJECTED])
def test_everything_else_is_rejected_with_a_reason(sql: str, reason: str) -> None:
    with pytest.raises(SqlRejected, match=reason):
        check_sql(sql)


def test_the_allowed_tables_are_exactly_what_the_agent_may_query(db: None) -> None:
    from app.db import reference

    assert set(reference.schema_columns()) == ALLOWED_TABLES


ROOT = Path(__file__).resolve().parent.parent.parent.parent


def owner_queries() -> list[str]:
    """Every gold query and example the owner wrote."""
    gold = [json.loads(line) for line in (ROOT / "evals/questions.jsonl").read_text().splitlines()]
    examples = [
        json.loads(line) for line in (ROOT / "evals/examples.jsonl").read_text().splitlines()
    ]
    return [q["gold_sql"] for q in gold if q.get("gold_sql")] + [e["sql"] for e in examples]


def test_every_gold_and_example_query_passes_the_checks() -> None:
    for sql in owner_queries():
        assert check_sql(limit_rows(check_sql(sql)))


def test_rebuilding_a_query_with_a_row_limit_does_not_change_its_result(db: None) -> None:
    from app.db.session import run_select

    queries = owner_queries()
    assert queries, "no gold or example queries to check"
    for sql in queries:
        assert run_select(limit_rows(sql))[1] == run_select(sql)[1], sql


# --- Row limit --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT ticker FROM companies", "SELECT ticker FROM companies LIMIT 500"),
        ("SELECT ticker FROM companies LIMIT 10", "SELECT ticker FROM companies LIMIT 10"),
        ("SELECT ticker FROM companies LIMIT 500", "SELECT ticker FROM companies LIMIT 500"),
        ("SELECT ticker FROM companies LIMIT 501", "SELECT ticker FROM companies LIMIT 500"),
        ("SELECT ticker FROM companies LIMIT 100000", "SELECT ticker FROM companies LIMIT 500"),
        ("SELECT ticker FROM companies LIMIT ALL", "SELECT ticker FROM companies LIMIT 500"),
        ("SELECT ticker FROM companies LIMIT 400 + 400", "SELECT ticker FROM companies LIMIT 500"),
        (
            "SELECT ticker FROM companies FETCH FIRST 9000 ROWS ONLY",
            "SELECT ticker FROM companies LIMIT 500",
        ),
        (
            "SELECT ticker FROM companies ORDER BY ticker LIMIT 900 OFFSET 20",
            "SELECT ticker FROM companies ORDER BY ticker LIMIT 500 OFFSET 20",
        ),
        (
            "SELECT ticker FROM companies ORDER BY ticker OFFSET 5",
            "SELECT ticker FROM companies ORDER BY ticker LIMIT 500 OFFSET 5",
        ),
        # Only the outermost statement is limited; an inner LIMIT is the query's own business.
        (
            "SELECT ticker FROM (SELECT ticker FROM companies LIMIT 9000) AS s",
            "SELECT ticker FROM (SELECT ticker FROM companies LIMIT 9000) AS s LIMIT 500",
        ),
        (
            "(SELECT ticker FROM companies) UNION ALL (SELECT ticker FROM dividends) ORDER BY 1",
            "(SELECT ticker FROM companies) UNION ALL (SELECT ticker FROM dividends) "
            "ORDER BY 1 LIMIT 500",
        ),
        # FETCH FIRST is LIMIT spelled out: a small count is kept, a large one is capped.
        (
            "SELECT ticker FROM companies ORDER BY ticker FETCH FIRST 5 ROWS ONLY",
            "SELECT ticker FROM companies ORDER BY ticker LIMIT 5",
        ),
        # Stray semicolons are not extra statements.
        ("SELECT ticker FROM companies;;", "SELECT ticker FROM companies LIMIT 500"),
        ("SELECT ticker FROM companies; ;", "SELECT ticker FROM companies LIMIT 500"),
        # Comments are dropped and a trailing semicolon goes.
        (
            "SELECT ticker /* why */ FROM companies -- LIMIT 5\n;",
            "SELECT ticker FROM companies LIMIT 500",
        ),
    ],
)
def test_limit_rows(sql: str, expected: str) -> None:
    assert limit_rows(sql) == expected
    assert limit_rows(expected) == expected  # stable when applied twice
    assert MAX_LIMIT == 500


# --- The node ---------------------------------------------------------------------------------


@pytest.fixture
def planned(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the database: record what was sent for planning and accept it."""
    sent: list[str] = []

    def run_select(sql: str, **kwargs: Any) -> tuple[list[str], list[Any], bool]:
        sent.append(sql)
        return ["QUERY PLAN"], [("Seq Scan",)], False

    monkeypatch.setattr(validate_module, "run_select", run_select)
    return sent


def test_a_valid_query_is_stored_row_limited_and_planned(planned: list[str]) -> None:
    update = validate_sql({"sql": "SELECT ticker FROM companies -- all of them"})

    assert set(update) == {"sql", "validation_error", "trace"}
    assert update["sql"] == "SELECT ticker FROM companies LIMIT 500"
    assert update["validation_error"] is None
    assert update["trace"][0]["note"] == "valid"
    # The database planned exactly the text that will run.
    assert planned == ["EXPLAIN SELECT ticker FROM companies LIMIT 500"]


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("DROP TABLE companies", "only SELECT queries are allowed, not DROP"),
        ("SELECT pg_sleep(60)", "the function pg_sleep is not allowed"),
        ("SELECT 1; SELECT 2", "2 statements were given; exactly one is allowed"),
        ("SELECT * FROM pg_user", "the table pg_user is not available"),
        ("", "no SQL statement was given"),
        (None, "no SQL statement was given"),
        (123, "the SQL is not text"),
        ("SELECT " + "(" * 5000 + "1" + ")" * 5000, "the SQL could not be parsed"),
    ],
)
def test_a_rejected_query_is_never_sent_to_the_database(
    planned: list[str], sql: Any, error: str
) -> None:
    update = validate_sql({"sql": sql})

    assert set(update) == {"validation_error", "trace"}  # `sql` is left as it was
    assert update["validation_error"].startswith(error)
    assert planned == []


@pytest.mark.parametrize("sql", ["SELECT 1; SELECT 2", "DROP TABLE x", "", "SELECT (((", 5])
def test_limit_rows_rejects_instead_of_crashing(sql: Any) -> None:
    with pytest.raises(SqlRejected, match="could not be rebuilt"):
        limit_rows(sql)


def test_stray_semicolons_do_not_crash_the_node(planned: list[str]) -> None:
    update = validate_sql({"sql": "SELECT ticker FROM companies;;"})

    assert update["sql"] == "SELECT ticker FROM companies LIMIT 500"
    assert update["validation_error"] is None


def test_validation_without_any_sql_in_the_state(planned: list[str]) -> None:
    assert validate_sql({})["validation_error"] == "no SQL statement was given"


def test_a_query_the_database_cannot_plan_is_a_validation_error(db: None) -> None:
    update = validate_sql({"sql": "SELECT revenu FROM income_statements WHERE ticker = 'AAPL'"})

    assert update["validation_error"] == (
        'the database could not plan the query: column "revenu" does not exist'
    )
    assert "sql" not in update


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ticker, revenue FROM income_statements WHERE fiscal_year = 2025",
        "WITH latest AS (SELECT ticker, MAX(period_end) AS d FROM balance_sheets GROUP BY ticker) "
        "SELECT b.ticker, b.total_assets FROM balance_sheets b "
        "JOIN latest l ON l.ticker = b.ticker AND l.d = b.period_end "
        "ORDER BY b.total_assets DESC LIMIT 5",
    ],
)
def test_a_good_query_passes_the_real_database_plan(db: None, sql: str) -> None:
    update = validate_sql({"sql": sql})

    assert update["validation_error"] is None
    assert update["sql"].endswith("LIMIT 500") or update["sql"].endswith("LIMIT 5")


def test_planning_does_not_run_the_query(db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.db.session.READER_STATEMENT_TIMEOUT_MS", 300)
    slow = "SELECT COUNT(*) FROM daily_prices a CROSS JOIN daily_prices b CROSS JOIN daily_prices c"

    assert validate_sql({"sql": slow})["validation_error"] is None  # planned at once, not run
