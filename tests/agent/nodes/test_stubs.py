"""Tests for the stub nodes of the graph skeleton: each returns only what it owns."""

from typing import Any, get_type_hints

import pytest

from app.agent.graph import NODES
from app.agent.nodes.answer import answer
from app.agent.nodes.repair_sql import repair_sql
from app.agent.nodes.validate_sql import validate_sql
from app.agent.state import AgentState

STATE_KEYS = set(get_type_hints(AgentState))

# The keys each node may write, besides `trace`.
OWNED = {
    "guardrail": {"refused", "refusal_reason"},
    "intent": {"intent"},
    "entities": {"tickers", "periods"},
    "retrieve_schema": {"schema_context", "examples"},
    "generate_sql": {"sql"},
    "validate_sql": {"validation_error"},
    "execute": {"columns", "rows", "db_error"},
    "repair_sql": {"sql", "validation_error", "db_error", "attempts"},
    "answer": {"answer"},
}
STATE: AgentState = {"question": "A question?", "sql": "SELECT 1", "attempts": 0, "trace": []}


@pytest.mark.parametrize("name", NODES)
def test_a_node_returns_only_its_own_keys_and_one_trace_event(name: str) -> None:
    update = NODES[name](STATE)

    assert set(update) - {"trace"} == OWNED[name]
    assert set(update) <= STATE_KEYS
    assert [event["node"] for event in update["trace"]] == [name]
    assert "question" not in update  # nodes never rewrite the question


def test_every_node_has_an_entry_above() -> None:
    assert set(OWNED) == set(NODES)


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("SELECT ticker FROM companies", None),
        ("DROP TABLE companies", "only SELECT queries are allowed, not DROP"),
        ("SELECT pg_sleep(60)", "the function pg_sleep is not allowed"),
        ("SELECT 1; SELECT 2", "2 statements were given; exactly one is allowed"),
        ("", "no SQL statement was given"),
        (None, "no SQL statement was given"),
        (123, "the SQL is not text"),
        ("SELECT " + "(" * 5000 + "1" + ")" * 5000, "the SQL could not be parsed"),
    ],
)
def test_validate_sql_node_records_why_a_query_was_rejected(sql: Any, error: str | None) -> None:
    update = validate_sql({"sql": sql})

    assert update["validation_error"] == error
    assert update["trace"][0]["note"] == ("valid" if error is None else f"rejected: {error}")


def test_validate_sql_node_without_any_sql_in_the_state() -> None:
    assert validate_sql({})["validation_error"] == "no SQL statement was given"


def test_repair_counts_each_attempt_and_clears_the_errors() -> None:
    failed: AgentState = {"sql": "SELECT x", "validation_error": "bad", "db_error": "boom"}

    first = repair_sql(failed)
    second = repair_sql(failed | {"attempts": 1})

    assert (first["attempts"], second["attempts"]) == (1, 2)
    assert (first["sql"], first["validation_error"], first["db_error"]) == ("SELECT x", None, None)


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
