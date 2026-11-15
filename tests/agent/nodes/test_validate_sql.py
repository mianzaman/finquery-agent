"""Tests for the SQL checks that need no database."""

import pytest

from app.agent.nodes.validate_sql import SqlRejected, check_sql

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
    "SELECT current_setting('statement_timeout')",
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
    ],
    ids=["number", "bytes", "list", "deep nesting", "very long"],
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
