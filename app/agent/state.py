"""AgentState: the data passed between graph nodes."""

import operator
import time
from typing import Annotated, Any, TypedDict


class TraceEvent(TypedDict):
    """One node's visit, for the evals and the UI to show what happened."""

    node: str
    duration_ms: float
    note: str


class AgentState(TypedDict, total=False):
    """Everything the graph knows about one question. A node returns only the keys it changes."""

    question: str

    # guardrail
    refused: bool
    refusal_reason: str | None

    # intent and entities
    intent: str
    tickers: list[str]
    periods: list[dict[str, Any]]

    # retrieve_schema
    schema_context: str
    examples: list[dict[str, str]]

    # generate_sql, validate_sql, execute, repair_sql
    sql: str | None
    validation_error: str | None
    db_error: str | None
    rows: list[tuple[Any, ...]]
    columns: list[str]
    attempts: int  # repairs made so far

    # answer
    answer: str

    # Appended to, never replaced: a node returns a list holding only its own event.
    trace: Annotated[list[TraceEvent], operator.add]


def trace_event(node: str, started: float, note: str = "") -> list[TraceEvent]:
    """The `trace` update for a node that began at `started` (a `time.perf_counter()` value)."""
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    return [TraceEvent(node=node, duration_ms=duration_ms, note=note)]
