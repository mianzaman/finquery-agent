"""Tests for the migration runner."""

import re
from pathlib import Path

import pytest

from db.migrate import (
    MIGRATIONS_DIR,
    MigrationError,
    migration_files,
    pending_files,
    reader_password,
)


def test_migration_files_are_numbered_and_ordered() -> None:
    names = [f.name for f in migration_files()]

    assert names == sorted(names)
    assert names[:4] == [
        "001_extensions.sql",
        "002_tables.sql",
        "003_vectors.sql",
        "004_reader_role.sql",
    ]
    assert all(re.fullmatch(r"\d{3}_[a-z0-9_]+\.sql", name) for name in names)


@pytest.mark.parametrize(
    ("applied", "expected"),
    [
        (set(), ["001_a.sql", "002_b.sql", "003_c.sql"]),
        ({"001_a.sql"}, ["002_b.sql", "003_c.sql"]),
        ({"001_a.sql", "002_b.sql", "003_c.sql"}, []),
        ({"999_unknown.sql"}, ["001_a.sql", "002_b.sql", "003_c.sql"]),
    ],
)
def test_pending_files(applied: set[str], expected: list[str]) -> None:
    files = [Path("001_a.sql"), Path("002_b.sql"), Path("003_c.sql")]

    assert [f.name for f in pending_files(files, applied)] == expected


def test_reader_password_is_taken_from_url() -> None:
    url = "postgresql+psycopg://agent_reader:p%40ss@localhost:5433/finquery"

    assert reader_password(url) == "p@ss"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://finquery:pw@localhost:5433/finquery",
        "postgresql+psycopg://agent_reader@localhost:5433/finquery",
        "postgresql+psycopg://agent_reader:@localhost:5433/finquery",
    ],
)
def test_reader_password_rejects_bad_urls(url: str) -> None:
    with pytest.raises(MigrationError) as excinfo:
        reader_password(url)

    assert "pw" not in str(excinfo.value)


def test_no_password_in_sql_files() -> None:
    for path in MIGRATIONS_DIR.glob("*.sql"):
        assert not re.search(r"password\s+'", path.read_text(), re.IGNORECASE), path.name


@pytest.mark.usefixtures("db")
def test_second_run_applies_nothing() -> None:
    from app.config import get_settings
    from app.db.session import admin_engine
    from db.migrate import run

    assert run(admin_engine, get_settings().reader_database_url.get_secret_value()) == []
