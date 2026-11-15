"""Tests for the agent state helpers."""

import time
from typing import get_type_hints

from app.agent.state import AgentState, trace_event

SPEC_KEYS = {
    "question", "refused", "refusal_reason", "intent", "tickers", "periods", "schema_context",
    "examples", "sql", "validation_error", "db_error", "rows", "columns", "attempts", "answer",
    "trace",
}  # fmt: skip


def test_state_has_exactly_the_agreed_keys() -> None:
    assert set(get_type_hints(AgentState)) == SPEC_KEYS


def test_trace_event_is_a_one_item_list_with_a_duration() -> None:
    started = time.perf_counter()

    (event,) = trace_event("intent", started, "lookup")

    assert (event["node"], event["note"]) == ("intent", "lookup")
    assert 0 <= event["duration_ms"] < 1000
    assert trace_event("intent", started)[0]["note"] == ""
