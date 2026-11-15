"""Retrieve-schema node: finds relevant schema docs and examples.

The question is embedded once and matched against the column descriptions and the example
questions (`app/db/retrieval.py`). The chosen tables become `schema_context`, a compact
CREATE-TABLE-like text; `companies` is always part of it. If the schema docs have not been
built yet, the context is the full schema without descriptions, so the agent still works.
"""

import time

from app.agent.state import AgentState, trace_event
from app.db import reference, retrieval
from app.db.retrieval import KEY_COLUMNS, TableDocs

SCHEMA_K = 12
EXAMPLES_K = 3


def format_table(docs: TableDocs, columns: tuple[tuple[str, str], ...]) -> str:
    """One table as text: its description, then the chosen columns with type and description.

    `columns` is the table's real column list. A described column that no longer exists is
    left out, and key columns appear even when nobody described them.
    """
    chosen = [
        (name, data_type)
        for name, data_type in columns
        if name in docs.columns or name in KEY_COLUMNS
    ]
    lines = [f"-- {docs.table}: {docs.description}" if docs.description else f"-- {docs.table}"]
    lines.append(f"{docs.table} (")
    for name, data_type in chosen:
        description = docs.columns.get(name)
        lines.append(f"  {name} {data_type}," + (f"  -- {description}" if description else ""))
    lines.append(")")
    return "\n".join(lines)


def full_schema(schema: dict[str, tuple[tuple[str, str], ...]]) -> str:
    """Every table with all its columns and no descriptions: the fallback context."""
    tables = [
        TableDocs(table, columns=dict.fromkeys(name for name, _ in columns))
        for table, columns in schema.items()
    ]
    return "\n\n".join(format_table(docs, schema[docs.table]) for docs in tables)


def retrieve_schema(state: AgentState) -> dict:
    """Choose the tables, columns and examples that fit the question."""
    started = time.perf_counter()
    question = state.get("question", "")
    schema = reference.schema_columns()
    tables = [docs for docs in retrieval.search_schema(question, SCHEMA_K) if docs.table in schema]
    described = [docs for docs in tables if docs.description or docs.columns]
    if not described:
        note = "no schema docs found; using the full schema"
        return {
            "schema_context": full_schema(schema),
            "examples": [],
            "trace": trace_event("retrieve_schema", started, note),
        }
    examples = retrieval.search_examples(question, EXAMPLES_K)
    context = "\n\n".join(format_table(docs, schema[docs.table]) for docs in tables)
    note = f"tables: {', '.join(docs.table for docs in tables)}; {len(examples)} examples"
    return {
        "schema_context": context,
        "examples": examples,
        "trace": trace_event("retrieve_schema", started, note),
    }
