"""pgvector queries for schema docs and query examples.

`search_schema` finds the columns whose descriptions are closest to a question and groups them
by table; `search_examples` finds the closest example questions with their SQL. Both run as the
read-only role through `run_select`, with fixed SQL and bound values.

The tables are filled by `python -m ingestion.build_embeddings`. Until that has run, both
searches return nothing and the caller falls back to the full schema.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.db.session import run_select
from app.llm import EMBEDDING_DIMENSIONS, get_embeddings

# Where the two tables live. Tests point this at a scratch schema so they never touch real rows.
RETRIEVAL_SCHEMA = "public"

# Columns that identify a row or its period. A table that is retrieved always comes with
# these, whichever of its columns matched, because a query cannot filter or join without them.
KEY_COLUMNS = (
    "ticker",
    "fiscal_year",
    "fiscal_quarter",
    "period_end",
    "trade_date",
    "ex_date",
    "split_date",
)


def schema_name() -> str:
    """`RETRIEVAL_SCHEMA`, checked to be a plain identifier because it is written into SQL."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", RETRIEVAL_SCHEMA):
        raise ValueError(f"RETRIEVAL_SCHEMA is not a plain identifier: {RETRIEVAL_SCHEMA!r}")
    return RETRIEVAL_SCHEMA


class RetrievalError(Exception):
    """The embedding model returned something unusable."""


@dataclass
class TableDocs:
    """One table chosen for a question: its description and the described columns to show."""

    table: str
    description: str = ""
    columns: dict[str, str] = field(default_factory=dict)  # column -> description
    distance: float = 2.0  # of its closest column; smaller is closer (cosine distance)


def vector_literal(vector: list[float]) -> str:
    """A vector as pgvector reads it: `[0.1,0.2,...]`."""
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise RetrievalError(
            f"the embedding has {len(vector)} dimensions, not {EMBEDDING_DIMENSIONS}; "
            "check EMBEDDING_PROVIDER"
        )
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


@lru_cache(maxsize=256)
def embed_question(question: str) -> str:
    """The question's embedding as a vector literal; cached, so two searches embed it once."""
    return vector_literal(get_embeddings().embed_query(question))


def search_schema(
    question: str, k: int = 12, always: tuple[str, ...] = ("companies",)
) -> list[TableDocs]:
    """The tables relevant to a question, closest first, then the `always` tables.

    The `k` closest column or table descriptions decide which tables are chosen. Each chosen
    table carries its own description, its matched columns and its key columns.
    """
    hits_sql = f"""
        SELECT table_name, column_name, description,
               embedding <=> CAST(%(vector)s AS vector) AS distance
        FROM {schema_name()}.schema_docs
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %(k)s
    """
    _, hits, _ = run_select(hits_sql, params={"vector": embed_question(question), "k": k})
    tables: dict[str, TableDocs] = {}
    for table, column, description, distance in hits:
        docs = tables.setdefault(table, TableDocs(table, distance=float(distance)))
        if column is None:
            docs.description = description
        else:
            docs.columns[column] = description
    for table in always:
        tables.setdefault(table, TableDocs(table))
    if not tables:
        return []

    # Add each chosen table's own description and its key columns.
    extras_sql = f"""
        SELECT table_name, column_name, description
        FROM {schema_name()}.schema_docs
        WHERE table_name = ANY(%(tables)s)
          AND (column_name IS NULL OR column_name = ANY(%(keys)s))
    """
    params = {"tables": list(tables), "keys": list(KEY_COLUMNS)}
    _, extras, _ = run_select(extras_sql, params=params)
    for table, column, description in extras:
        if column is None:
            tables[table].description = description
        else:
            tables[table].columns.setdefault(column, description)
    return sorted(tables.values(), key=lambda docs: docs.distance)


def search_examples(question: str, k: int = 3) -> list[dict[str, str]]:
    """The `k` example questions closest to this one, each with its SQL."""
    sql = f"""
        SELECT question, sql
        FROM {schema_name()}.query_examples
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(%(vector)s AS vector)
        LIMIT %(k)s
    """
    _, rows, _ = run_select(sql, params={"vector": embed_question(question), "k": k})
    return [{"question": question_text, "sql": sql_text} for question_text, sql_text in rows]
