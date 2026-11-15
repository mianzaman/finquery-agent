"""Applies the numbered SQL files in db/migrations."""

import logging
import sys
from pathlib import Path

from psycopg import sql
from sqlalchemy import Connection, Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
READER_ROLE = "agent_reader"

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """A migration could not be applied or the reader role could not be configured."""


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    """Return the migration files in the order they must run."""
    return sorted(directory.glob("*.sql"))


def pending_files(files: list[Path], applied: set[str]) -> list[Path]:
    """Return the files that are not recorded as applied, keeping their order."""
    return [f for f in files if f.name not in applied]


def reader_password(reader_url: str) -> str:
    """Take the reader role's password from its connection URL."""
    url = make_url(reader_url)
    if url.username != READER_ROLE:
        raise MigrationError(f"READER_DATABASE_URL must connect as {READER_ROLE}")
    if not url.password:
        raise MigrationError("READER_DATABASE_URL has no password")
    return url.password


def _applied(conn: Connection) -> set[str]:
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    )
    return set(conn.execute(text("SELECT filename FROM schema_migrations")).scalars())


def apply_migrations(engine: Engine, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every pending migration in one transaction and return the names applied."""
    with engine.begin() as conn:
        # One migrator at a time; released when the transaction ends.
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('schema_migrations'))"))
        todo = pending_files(migration_files(directory), _applied(conn))
        for path in todo:
            # Straight to psycopg with no parameters, so the file is sent as-is: several
            # statements are allowed and `%` is not read as a placeholder.
            conn.connection.driver_connection.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                text("INSERT INTO schema_migrations (filename) VALUES (:name)"),
                {"name": path.name},
            )
    return [path.name for path in todo]


def set_reader_password(engine: Engine, password: str) -> None:
    """Enable login for the reader role with the given password. Safe to repeat."""
    with engine.begin() as conn:
        raw = conn.connection.driver_connection
        # Hash on the client so the clear password never reaches the server or its logs.
        hashed = raw.pgconn.encrypt_password(
            password.encode(), READER_ROLE.encode(), b"scram-sha-256"
        ).decode()
        # ALTER ROLE takes no bind parameters; psycopg quotes the literal on the client.
        raw.execute(
            sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                sql.Identifier(READER_ROLE), sql.Literal(hashed)
            )
        )


def run(engine: Engine, reader_url: str) -> list[str]:
    """Bring the database up to date and return the names of the migrations applied."""
    password = reader_password(reader_url)
    applied = apply_migrations(engine)
    set_reader_password(engine, password)
    return applied


def main() -> int:
    """Command-line entry point: `uv run python -m db.migrate`."""
    from app.config import LOG_FORMAT, get_settings
    from app.db.session import admin_engine

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        applied = run(admin_engine, get_settings().reader_database_url.get_secret_value())
    except MigrationError as exc:
        logger.error("%s", exc)
        return 1
    except OperationalError:
        # The driver's message can include the connection target; keep it out of the log.
        logger.error("database unreachable; is `docker compose up -d db` running?")
        return 1
    for name in applied:
        logger.info("applied %s", name)
    if not applied:
        logger.info("nothing to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
