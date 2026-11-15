"""Validate-SQL node: every generated query must pass here before execution.

`check_sql` holds the checks that need no database: the text must be exactly one read-only
query. The graph node, the table allow-list, the LIMIT cap and the EXPLAIN step are added on top
of it in Task 09.

The database role is the main protection (it can only SELECT). These checks stop what the role
cannot: switching off its own statement timeout, sleeping, taking locks, and reading server files.
A deny-list can never be complete, so nothing here replaces the role: the first review found
two built-ins that run a SQL string (`ts_stat`, `ts_rewrite`). Task 09 should replace the
function deny-list with an allow-list.
"""

import time

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.agent.state import AgentState, trace_event

DIALECT = "postgres"
MAX_SQL_LENGTH = 20_000  # characters; far above any real query, and it bounds the parsing work

# Statements and clauses that write, lock or change session state, wherever they appear
# (a DELETE can hide inside a CTE). `Command` is what sqlglot returns for anything it does not
# parse fully: RESET, SHOW, EXPLAIN, CALL, DO, PREPARE, LOCK TABLE and so on.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Grant,
    exp.Command,
    exp.Set,
    exp.Into,  # SELECT ... INTO new_table
    exp.Lock,  # SELECT ... FOR UPDATE / FOR SHARE
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

# Functions a read-only role may still call but the agent never needs.
DENIED_FUNCTIONS = frozenset(
    {
        "set_config",  # switches off statement_timeout
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_notify",
        "pg_stat_file",
        "pg_export_snapshot",
        "nextval",
        "setval",
        "txid_current",
        # These run SQL given as a string, which would skip every check here. The text-search
        # pair is easy to miss: ts_stat('select ...') and ts_rewrite(query, 'select ...').
        "ts_stat",
        "ts_rewrite",
        "crosstab",  # tablefunc and xml2 extensions, in case one is ever installed
        "connectby",
        "xpath_table",
        # These build a string of any length in one call (up to 1 GB of server memory), and
        # the server does not check the statement timeout while it allocates.
        "repeat",
        "pad",  # sqlglot's name for lpad and rpad
        "lpad",
        "rpad",
    }
)
DENIED_FUNCTION_PREFIXES = (
    "pg_sleep",
    "pg_advisory",
    "pg_try_advisory",
    "pg_read_",  # pg_read_file, pg_read_binary_file, pg_read_server_files
    "pg_ls_",
    "pg_logical_",
    "pg_replication_",
    "lo_",  # large objects
    "dblink",
    # query_to_xml, cursor_to_xmlschema, table_to_xml, ...: the XML export family runs a query
    # given as a string, or dumps whole tables, schemas or the database.
    "query_to_",
    "cursor_to_",
    "table_to_",
    "schema_to_",
    "database_to_",
)


class SqlRejected(Exception):
    """The SQL is not a single read-only query. The message says why."""


def _function_name(node: exp.Func) -> str:
    # An unknown function keeps its own name; a known one reports sqlglot's canonical name.
    name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
    return name.lower()


def _is_denied(name: str) -> bool:
    return name in DENIED_FUNCTIONS or name.startswith(DENIED_FUNCTION_PREFIXES)


def check_sql(sql: str) -> str:
    """Return `sql` unchanged when it is one read-only query; otherwise raise `SqlRejected`."""
    if not isinstance(sql, str):
        raise SqlRejected("the SQL is not text")
    if len(sql) > MAX_SQL_LENGTH:
        raise SqlRejected(f"the SQL is longer than {MAX_SQL_LENGTH} characters")
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except (SqlglotError, RecursionError) as exc:
        # Deeply nested input can exhaust the parser's stack; that is a rejection, not a crash.
        raise SqlRejected("the SQL could not be parsed") from exc
    if not statements:
        raise SqlRejected("no SQL statement was given")
    if len(statements) > 1:
        raise SqlRejected(f"{len(statements)} statements were given; exactly one is allowed")

    statement = statements[0]
    if not isinstance(statement, exp.Select | exp.SetOperation | exp.Subquery):
        raise SqlRejected(f"only SELECT queries are allowed, not {statement.key.upper()}")
    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlRejected(f"{node.key.upper()} is not allowed in a query")
        if isinstance(node, exp.Func) and _is_denied(_function_name(node)):
            raise SqlRejected(f"the function {_function_name(node)} is not allowed")
    return sql


def validate_sql(state: AgentState) -> dict:
    """Graph node: put the generated SQL through `check_sql` and record why it failed, if it did.

    Unlike the other Task 06 nodes this is not a stub: the check needs no database or model, so
    the safety gate is in the graph from the start. Task 09 adds the remaining checks.
    """
    started = time.perf_counter()
    sql = state.get("sql")
    try:
        if sql is None:
            raise SqlRejected("no SQL statement was given")
        check_sql(sql)
    except SqlRejected as exc:
        return {
            "validation_error": str(exc),
            "trace": trace_event("validate_sql", started, f"rejected: {exc}"),
        }
    return {"validation_error": None, "trace": trace_event("validate_sql", started, "valid")}
