"""What the API routes call: runs the agent and shapes its result for the API contract.

The routes in `app/main.py` hold no logic; this module turns the agent's final state into an
`AskResponse` and the owner's schema descriptions into a `SchemaResponse`.
"""

import math
import time
from datetime import date
from decimal import Decimal
from functools import lru_cache
from typing import Any

from app.agent.graph import run_agent
from app.schemas import AskResponse, ColumnDoc, SchemaResponse, TableDoc
from ingestion.build_embeddings import read_schema_docs


def json_value(value: Any) -> Any:
    """A cell as JSON: numbers stay numbers, dates become ISO text, NaN becomes null."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, date):  # a datetime is a date too
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    return str(value)


def ask(question: str, request_id: str) -> AskResponse:
    """Run one question through the agent. Model failures are raised for the API to report."""
    started = time.perf_counter()
    state = run_agent(question)
    refused = bool(state.get("refused"))
    checked = "unsupported_numbers" in state
    return AskResponse(
        answer=None if refused else state.get("answer"),
        sql=None if refused else state.get("sql"),
        columns=[] if refused else list(state.get("columns") or []),
        rows=[] if refused else [json_value(row) for row in state.get("rows") or []],
        refused=refused,
        refusal_reason=state.get("refusal_reason") if refused else None,
        attempts=state.get("attempts", 0),
        faithfulness_ok=not state.get("faithfulness_failed", False) if checked else None,
        latency_ms=round((time.perf_counter() - started) * 1000),
        request_id=request_id,
    )


@lru_cache
def schema() -> SchemaResponse:
    """The tables and columns the agent can query, with the owner's descriptions."""
    tables: dict[str, TableDoc] = {}
    for doc in read_schema_docs():
        if doc.column is None:
            tables[doc.table] = TableDoc(name=doc.table, description=doc.description, columns=[])
        else:
            tables[doc.table].columns.append(
                ColumnDoc(name=doc.column, description=doc.description)
            )
    return SchemaResponse(tables=list(tables.values()))
