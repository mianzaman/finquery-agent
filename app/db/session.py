"""Database engines: admin_engine and reader_engine."""

from typing import Any

import psycopg
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError

from app.config import get_settings

READER_STATEMENT_TIMEOUT_MS = 5000
MAX_ROWS = 5000


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


class QueryError(Exception):
    """A query failed on the read-only engine. The message is the database's own, first line."""


def run_select(
    sql: str,
    max_rows: int = MAX_ROWS,
    engine: Engine | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Run one validated query as `agent_reader` and return `(columns, rows, truncated)`.

    Callers pass the SQL through `check_sql` first. This function adds what the database itself
    can enforce:
    - a read-only transaction, on top of the role's SELECT-only grants;
    - the statement timeout set again for this transaction, because a session-level `SET` from
      an earlier query would survive on a pooled connection;
    - the extended query protocol (`prepare=True`), under which the server refuses a string
      that holds more than one statement, whatever the SQL parser made of it.
    At most `max_rows` rows are fetched; `truncated` says whether more were available.

    `params` is for the application's own fixed queries (`%(name)s` placeholders), never for
    model-written SQL: with it, a literal `%` in the SQL must be written `%%`.
    """
    try:
        connection = (engine or reader_engine).connect()
    except OperationalError as exc:
        # The driver's message can include the connection target; keep it out of the error.
        raise QueryError("the database is unreachable") from exc
    with connection as conn:
        raw = conn.connection.driver_connection
        try:
            with raw.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(f"SET LOCAL statement_timeout = {READER_STATEMENT_TIMEOUT_MS}")
                cursor.execute(sql, params, prepare=True)
                if cursor.description is None:
                    raise QueryError("the statement returned no result set")
                columns = [column.name for column in cursor.description]
                rows = cursor.fetchmany(max_rows + 1)
        except psycopg.Error as exc:
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            raise QueryError(message) from exc
        except UnicodeError as exc:
            # Text the driver cannot encode (a lone surrogate) never reaches the server.
            raise QueryError("the query contains characters that cannot be sent") from exc
        finally:
            raw.rollback()
    return columns, rows[:max_rows], len(rows) > max_rows
