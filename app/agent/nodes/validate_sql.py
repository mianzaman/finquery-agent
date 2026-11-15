"""Validate-SQL node: every generated query must pass here before execution.

Three layers, in this order:

1. `check_sql` needs no database. The text must be exactly one read-only query that reads only
   the allowed tables and calls only the allowed functions. The baseline and the eval's gold
   SQL go through it too.
2. `limit_rows` caps the result at `MAX_LIMIT` rows and rebuilds the statement from its parsed
   form, without comments. What it returns is the text that is stored in the state and later
   executed: nothing runs that was not validated in exactly that form.
3. `EXPLAIN` on the read-only engine catches what only the database knows (an unknown column,
   a type mismatch), so the query is repaired without ever running.

The database role is still the main protection (it can only SELECT). These checks stop what the
role cannot: switching off its own statement timeout, sleeping, taking locks, reading server
catalogs and settings, and running SQL hidden in a string. Functions are on an allow-list
because a deny-list missed two built-ins that execute a SQL string (`ts_stat`, `ts_rewrite`);
the deny-list is kept behind it as a second check.
"""

import time

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from app.agent.state import AgentState, trace_event
from app.db.session import QueryError, run_select

DIALECT = "postgres"
MAX_SQL_LENGTH = 20_000  # characters; far above any real query, and it bounds the parsing work
MAX_LIMIT = 500  # rows a query may return

# The only relations a query may read: the data tables and the split-adjusted view. Everything
# else the role can see (server catalogs, information_schema, the retrieval tables) is refused.
ALLOWED_TABLES = frozenset(
    {
        "companies",
        "income_statements",
        "income_statements_split_adjusted",
        "balance_sheets",
        "cash_flows",
        "daily_prices",
        "dividends",
        "stock_splits",
    }
)
ALLOWED_SCHEMAS = frozenset({"", "public"})

# The only functions a query may call, by the name sqlglot gives them (it renames some:
# string_agg is group_concat, to_char is time_to_str, bool_and is logical_and, date_trunc is
# timestamp_trunc). All are pure calculations on the query's own rows. A function that is not
# here is rejected by name, so the model can rewrite the query without it.
ALLOWED_FUNCTIONS = frozenset(
    {
        # aggregates
        "count", "sum", "avg", "min", "max", "median", "mode", "stddev", "stddev_pop",
        "stddev_samp", "variance", "variance_pop", "var_samp", "var_pop", "corr",
        "percentile_cont", "percentile_disc", "any_value", "logical_and", "logical_or",
        "bool_and", "bool_or", "array_agg", "group_concat", "string_agg",
        # window
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead",
        "first_value", "last_value", "nth_value",
        # conditional and logical
        "case", "coalesce", "nullif", "greatest", "least", "if", "and", "or", "xor", "exists",
        # casts
        "cast", "try_cast",
        # math
        "abs", "round", "ceil", "ceiling", "floor", "trunc", "sign", "sqrt", "power", "pow",
        "exp", "ln", "log", "mod", "div",
        # dates and times
        "extract", "date_part", "timestamp_trunc", "date_trunc", "date", "make_date", "age",
        "current_date", "current_timestamp", "current_time", "now", "time_to_str", "to_char",
        "str_to_date", "to_date", "str_to_time", "to_timestamp", "date_add", "date_sub",
        "ts_or_ds_to_date", "ts_or_ds_to_timestamp",
        # text
        "upper", "lower", "initcap", "length", "char_length", "trim", "ltrim", "rtrim",
        "left", "right", "substring", "substr", "concat", "concat_ws", "split_part",
        "str_position", "strpos", "position",
        # small sets of values (for `= ANY(ARRAY[...])`)
        "array", "unnest", "explode",
    }
)  # fmt: skip
# Deliberately absent: `replace` (nested calls build 300 MB strings in a second) and
# `generate_series` (an endless row source for string_agg and array_agg). With neither, a query
# can only grow from the rows of the data tables, which the statement timeout bounds.

# Keywords that read like columns but return the session's role or database. The role's name is
# not secret, but a query has no business with it.
SESSION_KEYWORDS = frozenset(
    {
        "user", "current_user", "current_role", "session_user", "system_user",
        "current_catalog", "current_schema", "current_database",
    }
)  # fmt: skip

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


def _real_tables(statement: exp.Expression) -> list[exp.Table]:
    """Every table reference that is not a reference to one of the query's own CTEs.

    A CTE can be named like a real table, and only inside its own scope does the name mean the
    CTE. sqlglot's scope analysis tells the two apart; a reference it does not classify is
    treated as a real table, which is the safe direction.
    """
    cte_references: set[int] = set()
    for scope in traverse_scope(statement):
        for table in scope.tables:
            if table.name in scope.cte_sources and not table.db and not table.catalog:
                cte_references.add(id(table))
    return [table for table in statement.find_all(exp.Table) if id(table) not in cte_references]


def _check_tables(statement: exp.Expression) -> None:
    for table in _real_tables(statement):
        if isinstance(table.this, exp.Func):
            continue  # a function used as a table, such as unnest(): checked as a function
        if not isinstance(table.this, exp.Identifier):
            raise SqlRejected("a table name in the query could not be understood")
        name = table.name.lower()
        if table.catalog or table.db.lower() not in ALLOWED_SCHEMAS or name not in ALLOWED_TABLES:
            shown = ".".join(part for part in (table.catalog, table.db, table.name) if part)
            raise SqlRejected(
                f"the table {shown} is not available; use only {', '.join(sorted(ALLOWED_TABLES))}"
            )


def check_sql(sql: str) -> str:
    """Return `sql` unchanged when it is one read-only query on the allowed tables and functions.

    Otherwise raise `SqlRejected` with a reason the model can act on.
    """
    if not isinstance(sql, str):
        raise SqlRejected("the SQL is not text")
    if len(sql) > MAX_SQL_LENGTH:
        raise SqlRejected(f"the SQL is longer than {MAX_SQL_LENGTH} characters")
    try:
        sql.encode("utf-8")
    except UnicodeError as exc:
        raise SqlRejected("the SQL contains characters that are not valid text") from exc
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
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if _is_denied(name) or name not in ALLOWED_FUNCTIONS:
                raise SqlRejected(f"the function {name} is not allowed")
        if isinstance(node, exp.Column) and not node.table and not node.this.quoted:
            if node.name.lower() in SESSION_KEYWORDS:
                raise SqlRejected(f"{node.name} is not allowed in a query")
        if isinstance(node, exp.Cast) and node.to.sql(dialect=DIALECT).lower().startswith("reg"):
            # regclass, regproc, regrole...: casts that look names up in the server catalogs
            raise SqlRejected(f"a cast to {node.to.sql(dialect=DIALECT).lower()} is not allowed")
    try:
        _check_tables(statement)
    except (SqlglotError, RecursionError) as exc:
        raise SqlRejected("the SQL could not be analysed") from exc
    return sql


def limit_rows(sql: str, limit: int = MAX_LIMIT) -> str:
    """Return checked SQL rebuilt so that it returns at most `limit` rows.

    A query with no LIMIT gets one; a LIMIT above the cap, `LIMIT ALL`, `FETCH FIRST` or a
    computed LIMIT is replaced by the cap. A smaller plain LIMIT is kept, and so is OFFSET.
    Only the outermost statement is touched. Comments are dropped, since the statement is
    written out again from its parsed form.
    """
    try:
        # Parsed the same way as in check_sql, so both see the same single statement.
        (statement,) = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
        clause = statement.args.get("limit")
        count = None
        if isinstance(clause, exp.Limit):
            count = clause.expression
        elif isinstance(clause, exp.Fetch):  # FETCH FIRST n ROWS: the same thing, spelled out
            count = clause.args.get("count")
        small = isinstance(count, exp.Literal) and count.is_int and 0 <= int(count.this) <= limit
        if not small:
            statement = statement.limit(limit, copy=False)
        elif isinstance(clause, exp.Fetch):
            statement = statement.limit(int(count.this), copy=False)
        return statement.sql(dialect=DIALECT, comments=False)
    except (SqlglotError, RecursionError, ValueError, AttributeError, TypeError) as exc:
        raise SqlRejected("the SQL could not be rebuilt with a row limit") from exc


def validate_sql(state: AgentState) -> dict:
    """Graph node: check the SQL, cap its rows, and ask the database to plan it.

    On success `sql` is replaced by the checked, row-limited text, which is exactly what
    `execute` will run. On failure `validation_error` says why, for the repair node.
    """
    started = time.perf_counter()
    sql = state.get("sql")
    try:
        if sql is None:
            raise SqlRejected("no SQL statement was given")
        limited = limit_rows(check_sql(sql))
        check_sql(limited)  # the rebuilt text is what runs, so it is the text that must pass
        run_select(f"EXPLAIN {limited}")
    except SqlRejected as exc:
        return {
            "validation_error": str(exc),
            "trace": trace_event("validate_sql", started, f"rejected: {exc}"),
        }
    except QueryError as exc:
        return {
            "validation_error": f"the database could not plan the query: {exc}",
            "trace": trace_event("validate_sql", started, f"rejected by EXPLAIN: {exc}"),
        }
    return {
        "sql": limited,
        "validation_error": None,
        "trace": trace_event("validate_sql", started, "valid"),
    }
