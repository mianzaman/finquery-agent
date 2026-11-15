"""Retrieve-schema node: finds relevant schema docs and examples.

The question is embedded once and matched against the column descriptions and the example
questions (`app/db/retrieval.py`). The search decides which tables matter; each chosen table is
then shown whole, with every column, its type and its description, as CREATE-TABLE-like text.
`companies` is always part of it. If the schema docs have not been built yet, the context is
the full schema without descriptions, so the agent still works.
"""

import time

from app.agent.state import AgentState, trace_event
from app.db import reference, retrieval

SCHEMA_K = 12
EXAMPLES_K = 3

Columns = tuple[tuple[str, str], ...]


def format_table(table: str, columns: Columns, docs: dict[str | None, str]) -> str:
    """One table as text: its description, then every real column with type and description.

    `columns` is the table's real column list, so a described column that no longer exists
    is left out. `docs` maps a column name to its description (None for the table itself).
    """
    description = docs.get(None)
    lines = [f"-- {table}: {description}" if description else f"-- {table}", f"{table} ("]
    for name, data_type in columns:
        note = docs.get(name)
        lines.append(f"  {name} {data_type}," + (f"  -- {note}" if note else ""))
    lines.append(")")
    return "\n".join(lines)


def full_schema(schema: dict[str, Columns]) -> str:
    """Every table with all its columns and no descriptions: the fallback context."""
    return "\n\n".join(format_table(table, columns, {}) for table, columns in schema.items())


def retrieve_schema(state: AgentState) -> dict:
    """Choose the tables and examples that fit the question."""
    started = time.perf_counter()
    question = state.get("question", "")
    schema = reference.schema_columns()
    found = retrieval.search_schema(question, SCHEMA_K)
    tables = [docs.table for docs in found if docs.table in schema]
    if not any(docs.description or docs.columns for docs in found):
        note = "no schema docs found; using the full schema"
        return {
            "schema_context": full_schema(schema),
            "examples": [],
            "trace": trace_event("retrieve_schema", started, note),
        }
    docs = retrieval.table_docs(tables)
    examples = retrieval.search_examples(question, EXAMPLES_K)
    context = "\n\n".join(
        format_table(table, schema[table], docs.get(table, {})) for table in tables
    )
    note = f"tables: {', '.join(tables)}; {len(examples)} examples"
    return {
        "schema_context": context,
        "examples": examples,
        "trace": trace_event("retrieve_schema", started, note),
    }
