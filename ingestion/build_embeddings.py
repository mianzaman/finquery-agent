"""Builds embeddings for schema docs and query examples.

Run with `uv run python -m ingestion.build_embeddings [--rebuild]`.

It reads the owner's two files and fills the retrieval tables:
- `db/schema_docs.yaml` -> `schema_docs`. One row per table (its description, with
  `column_name` NULL) and one per described column. The text that is embedded is
  `table: description` or `table.column: description`.
- `evals/examples.jsonl` -> `query_examples`. One row per example; the question is embedded.

Only rows whose text changed are embedded again (a SHA-256 of the text is stored). Rows that
are no longer in the files are removed. Vectors from two embedding models are never mixed:
after changing `EMBEDDING_PROVIDER`, run with `--rebuild` to embed everything again.

The files are never written here. The script also reports tables and columns the agent can
query that have no description, and described ones that do not exist.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import OperationalError

from app.config import LOG_FORMAT
from app.db import retrieval
from app.db.retrieval import vector_literal

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DOCS_PATH = ROOT / "db" / "schema_docs.yaml"
EXAMPLES_PATH = ROOT / "evals" / "examples.jsonl"
EMBED_BATCH = 64

logger = logging.getLogger(__name__)

Embed = Callable[[list[str]], list[list[float]]]


class BuildError(Exception):
    """The embeddings cannot be built as asked. The message says what to change."""


@dataclass(frozen=True)
class SchemaDoc:
    """One description: of a table (`column` None) or of one of its columns."""

    table: str
    column: str | None
    description: str

    @property
    def text(self) -> str:
        name = self.table if self.column is None else f"{self.table}.{self.column}"
        return f"{name}: {self.description}"


@dataclass(frozen=True)
class Example:
    """One example question with the SQL that answers it."""

    question: str
    sql: str


# --- Reading the owner's files ----------------------------------------------------------------


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


def read_schema_docs(path: Path = SCHEMA_DOCS_PATH) -> list[SchemaDoc]:
    """Read `tables -> <table> -> description, columns -> <column>: description`."""
    if not path.exists():
        raise BuildError(f"{path} does not exist")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tables = loaded.get("tables") if isinstance(loaded, dict) else None
    if not isinstance(tables, dict) or not tables:
        raise BuildError(
            f"{path.name} has no descriptions yet. Expected: "
            "tables -> <table> -> description, columns -> <column>: description"
        )
    docs: list[SchemaDoc] = []
    for table, body in tables.items():
        if not isinstance(body, dict):
            raise BuildError(f"{path.name}: {table} must have `description` and `columns`")
        description = _one_line(body.get("description") or "")
        if not description:
            raise BuildError(f"{path.name}: {table} has no description")
        docs.append(SchemaDoc(str(table), None, description))
        columns = body.get("columns") or {}
        if not isinstance(columns, dict):
            raise BuildError(f"{path.name}: {table}.columns must map column names to text")
        for column, column_description in columns.items():
            if not _one_line(column_description or ""):
                raise BuildError(f"{path.name}: {table}.{column} has no description")
            docs.append(SchemaDoc(str(table), str(column), _one_line(column_description)))
    return docs


def read_examples(path: Path = EXAMPLES_PATH) -> list[Example]:
    """Read one JSON object per line with `question` and `sql` (an `id` is allowed)."""
    if not path.exists():
        raise BuildError(f"{path} does not exist")
    examples: list[Example] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildError(f"{where} is not valid JSON") from exc
        question, sql = _one_line(raw.get("question") or ""), str(raw.get("sql") or "").strip()
        if not question or not sql:
            raise BuildError(f"{where} needs a question and its sql")
        examples.append(Example(question, sql))
    repeated = {e.question for e in examples if sum(o.question == e.question for o in examples) > 1}
    if repeated:
        raise BuildError(f"{path.name} repeats the question: {sorted(repeated)[0]}")
    return examples


def content_hash(embedded_text: str) -> str:
    """SHA-256 of the text an embedding was made from."""
    return hashlib.sha256(embedded_text.encode("utf-8")).hexdigest()


# --- Writing the retrieval tables -------------------------------------------------------------


def _embed_all(embed: Embed, texts: list[str]) -> list[str]:
    """Embed texts in batches and return pgvector literals; the size of each is checked."""
    literals: list[str] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vectors = embed(batch)
        if len(vectors) != len(batch):
            raise BuildError(f"the model returned {len(vectors)} vectors for {len(batch)} texts")
        try:
            literals += [vector_literal(list(vector)) for vector in vectors]
        except retrieval.RetrievalError as exc:
            raise BuildError(str(exc)) from exc
    return literals


def check_model(conn: Connection, model: str, rebuild: bool) -> None:
    """Refuse to add vectors next to ones made by a different embedding model."""
    schema = retrieval.schema_name()
    found: set[str] = set()
    for table in ("schema_docs", "query_examples"):
        query = text(
            f"SELECT DISTINCT embedding_model FROM {schema}.{table} WHERE embedding IS NOT NULL"
        )
        found |= {name for name in conn.execute(query).scalars() if name}
    others = sorted(found - {model})
    if others and not rebuild:
        raise BuildError(
            f"the tables hold embeddings from {', '.join(others)}, but the model is now {model}. "
            "Run again with --rebuild to embed everything with the new model."
        )


def sync_schema_docs(
    conn: Connection, docs: list[SchemaDoc], embed: Embed, model: str, rebuild: bool = False
) -> dict[str, int]:
    """Make `schema_docs` match the file. Returns counts: embedded, unchanged, removed."""
    schema = retrieval.schema_name()
    stored = {
        (row.table_name, row.column_name): (row.content_hash, row.embedding_model)
        for row in conn.execute(
            text(
                f"SELECT table_name, column_name, content_hash, embedding_model "
                f"FROM {schema}.schema_docs"
            )
        )
    }
    wanted = {(doc.table, doc.column): doc for doc in docs}
    changed = [
        doc
        for key, doc in wanted.items()
        if rebuild or stored.get(key) != (content_hash(doc.text), model)
    ]
    upsert = text(
        f"""
        INSERT INTO {schema}.schema_docs
            (table_name, column_name, description, embedding, embedding_model, content_hash)
        VALUES (:table, :column, :description, CAST(:embedding AS vector), :model, :hash)
        ON CONFLICT (table_name, column_name) DO UPDATE SET
            description = EXCLUDED.description,
            embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model,
            content_hash = EXCLUDED.content_hash
        """
    )
    for doc, embedding in zip(changed, _embed_all(embed, [d.text for d in changed]), strict=True):
        values = {
            "table": doc.table,
            "column": doc.column,
            "description": doc.description,
            "embedding": embedding,
            "model": model,
            "hash": content_hash(doc.text),
        }
        conn.execute(upsert, values)
    removed = [key for key in stored if key not in wanted]
    delete = text(
        f"DELETE FROM {schema}.schema_docs "
        "WHERE table_name = :table AND column_name IS NOT DISTINCT FROM :column"
    )
    for table, column in removed:
        conn.execute(delete, {"table": table, "column": column})
    return {
        "embedded": len(changed),
        "unchanged": len(wanted) - len(changed),
        "removed": len(removed),
    }


def sync_examples(
    conn: Connection, examples: list[Example], embed: Embed, model: str, rebuild: bool = False
) -> dict[str, int]:
    """Make `query_examples` match the file. A changed SQL alone needs no new embedding."""
    schema = retrieval.schema_name()
    stored = {
        row.question: (row.content_hash, row.embedding_model, row.sql)
        for row in conn.execute(
            text(
                f"SELECT question, sql, content_hash, embedding_model FROM {schema}.query_examples"
            )
        )
    }
    wanted = {example.question: example for example in examples}
    changed = [
        example
        for question, example in wanted.items()
        if rebuild or stored.get(question, (None, None))[:2] != (content_hash(question), model)
    ]
    upsert = text(
        f"""
        INSERT INTO {schema}.query_examples
            (question, sql, embedding, embedding_model, content_hash)
        VALUES (:question, :sql, CAST(:embedding AS vector), :model, :hash)
        ON CONFLICT (question) DO UPDATE SET
            sql = EXCLUDED.sql,
            embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model,
            content_hash = EXCLUDED.content_hash
        """
    )
    embeddings = _embed_all(embed, [example.question for example in changed])
    for example, embedding in zip(changed, embeddings, strict=True):
        values = {
            "question": example.question,
            "sql": example.sql,
            "embedding": embedding,
            "model": model,
            "hash": content_hash(example.question),
        }
        conn.execute(upsert, values)
    fresh = {example.question for example in changed}
    sql_only = [
        example
        for question, example in wanted.items()
        if question not in fresh and stored[question][2] != example.sql
    ]
    for example in sql_only:
        conn.execute(
            text(f"UPDATE {schema}.query_examples SET sql = :sql WHERE question = :question"),
            {"sql": example.sql, "question": example.question},
        )
    removed = [question for question in stored if question not in wanted]
    for question in removed:
        conn.execute(
            text(f"DELETE FROM {schema}.query_examples WHERE question = :question"),
            {"question": question},
        )
    return {
        "embedded": len(changed),
        "unchanged": len(wanted) - len(changed) - len(sql_only),
        "sql_updated": len(sql_only),
        "removed": len(removed),
    }


def coverage(
    docs: list[SchemaDoc], schema: dict[str, tuple[tuple[str, str], ...]]
) -> dict[str, list[str]]:
    """Compare the descriptions with the real schema.

    `undescribed`: tables and columns a query can use that nobody described.
    `unknown`: described tables and columns that do not exist (a typo, or a dropped column).
    A view's column may be described on the table it comes from, so a column counts as
    described when any table describes that name.
    """
    described_tables = {doc.table for doc in docs}
    described_columns = {(doc.table, doc.column) for doc in docs if doc.column}
    described_names = {doc.column for doc in docs if doc.column}
    undescribed = [table for table in schema if table not in described_tables]
    undescribed += [
        f"{table}.{column}"
        for table, columns in schema.items()
        for column, _ in columns
        if (table, column) not in described_columns and column not in described_names
    ]
    unknown = [table for table in sorted(described_tables) if table not in schema]
    unknown += [
        f"{table}.{column}"
        for table, column in sorted(described_columns)
        if table in schema and column not in {name for name, _ in schema[table]}
    ]
    return {"undescribed": undescribed, "unknown": unknown}


def build(
    engine: Engine,
    embed: Embed,
    model: str,
    docs: list[SchemaDoc],
    examples: list[Example],
    rebuild: bool = False,
    prune: bool = False,
) -> dict[str, dict[str, int]]:
    """Fill both tables in one transaction and return the counts for each.

    Removing more than half of a table's rows is refused unless `prune` is set: that is what a
    truncated or emptied file looks like, and nothing is changed when it happens.
    """
    with engine.begin() as conn:
        check_model(conn, model, rebuild)
        counts = {
            "schema_docs": sync_schema_docs(conn, docs, embed, model, rebuild),
            "query_examples": sync_examples(conn, examples, embed, model, rebuild),
        }
        for table, numbers in counts.items():
            kept = sum(count for name, count in numbers.items() if name != "removed")
            if not prune and numbers["removed"] > kept:
                raise BuildError(
                    f"{numbers['removed']} of {numbers['removed'] + kept} rows would be removed "
                    f"from {table}, which looks like a truncated file. Nothing was changed; "
                    "run again with --prune if this is intended."
                )
        return counts


# --- Command line -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    from app.config import get_settings
    from app.db import reference
    from app.db.session import QueryError, admin_engine
    from app.llm import LLMError, embedding_model_name, get_embeddings

    parser = argparse.ArgumentParser(description="Embed the schema docs and query examples.")
    parser.add_argument("--rebuild", action="store_true", help="embed every row again")
    parser.add_argument(
        "--prune", action="store_true", help="allow removing more than half of a table's rows"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    try:
        docs = read_schema_docs()
        examples = read_examples()
        if not examples:
            logger.warning("%s has no examples; none will be retrieved", EXAMPLES_PATH.name)
        model = embedding_model_name(get_settings())
        logger.info(
            "embedding %d descriptions and %d examples with %s", len(docs), len(examples), model
        )
        embed = get_embeddings().embed_documents
        counts = build(admin_engine, embed, model, docs, examples, args.rebuild, args.prune)
        gaps = coverage(docs, reference.schema_columns())
    except (BuildError, LLMError, QueryError) as exc:
        logger.error("%s", exc)
        return 1
    except OperationalError:
        # The driver's message can include the connection target; keep it out of the log.
        logger.error("database unreachable; is `docker compose up -d db` running?")
        return 1

    for table, numbers in counts.items():
        summary = ", ".join(f"{count} {name}" for name, count in numbers.items())
        logger.info("%s: %s", table, summary)
    if gaps["undescribed"]:
        logger.warning("no description for: %s", ", ".join(gaps["undescribed"]))
    if gaps["unknown"]:
        logger.warning("described but not in the database: %s", ", ".join(gaps["unknown"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
