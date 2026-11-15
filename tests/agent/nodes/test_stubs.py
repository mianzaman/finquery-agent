"""Tests for the one node that is still a stub: `answer` (real logic comes in Task 10)."""

from typing import get_type_hints

import pytest

from app.agent.graph import NODES
from app.agent.nodes.answer import answer
from app.agent.state import AgentState

STATE_KEYS = set(get_type_hints(AgentState))


def test_the_stub_returns_only_its_own_key_and_one_trace_event() -> None:
    update = answer({"question": "A question?", "rows": [(1,)]})

    assert set(update) == {"answer", "trace"}
    assert set(update) <= STATE_KEYS
    assert [event["node"] for event in update["trace"]] == ["answer"]


def test_every_node_is_wired() -> None:
    assert list(NODES) == [
        "guardrail",
        "intent",
        "entities",
        "retrieve_schema",
        "generate_sql",
        "validate_sql",
        "execute",
        "repair_sql",
        "answer",
    ]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"rows": [(1,), (2,)]}, "Stub answer: the query returned 2 rows."),
        ({}, "Stub answer: the query returned 0 rows."),
        (
            {"validation_error": "bad SQL", "attempts": 2},
            "I could not answer this: the query still failed after 2 repairs (bad SQL).",
        ),
        (
            {"db_error": "timeout", "attempts": 2, "rows": [(1,)]},
            "I could not answer this: the query still failed after 2 repairs (timeout).",
        ),
    ],
)
def test_answer_reports_rows_or_the_failure(state: AgentState, expected: str) -> None:
    assert answer(state)["answer"] == expected
