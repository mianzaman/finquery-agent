"""Route tests for the agent graph: nodes are replaced with fakes, so no LLM or database is used."""

from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END

from app.agent.graph import (
    MAX_REPAIRS,
    NODES,
    build_graph,
    initial_state,
    route_after_execute,
    route_after_guardrail,
    route_after_validation,
    run_agent,
)
from app.agent.state import AgentState, trace_event

GRAPH_FILE = Path(__file__).parent.parent.parent / "docs" / "graph.mmd"
TO_VALIDATION = [
    "guardrail",
    "intent",
    "entities",
    "retrieve_schema",
    "generate_sql",
    "validate_sql",
]
REPAIR = ["repair_sql", "validate_sql"]


def fake(name: str, *updates: dict[str, Any]) -> Any:
    """A node that returns the given updates in turn (the last one repeats) and records a visit."""
    calls = iter(range(10_000))

    def node(state: AgentState) -> dict[str, Any]:
        update = updates[min(next(calls), len(updates) - 1)]
        return update | {"trace": trace_event(name, 0.0, "fake")}

    return node


def visited(state: AgentState) -> list[str]:
    return [event["node"] for event in state["trace"]]


INVALID = {"validation_error": "only SELECT queries are allowed"}
VALID = {"validation_error": None}
DB_ERROR = {"db_error": 'column "nope" does not exist'}
DB_OK = {"db_error": None, "columns": ["value"], "rows": [(1,)]}


def run(**overrides: Any) -> AgentState:
    return build_graph(overrides).invoke(initial_state("What was Apple's revenue?"))


def test_happy_path_visits_every_node_in_order() -> None:
    state = run_agent("What was Apple's revenue?")

    assert visited(state) == [*TO_VALIDATION, "execute", "answer"]
    assert state["question"] == "What was Apple's revenue?"
    assert (state["refused"], state["attempts"]) == (False, 0)
    assert (state["sql"], state["validation_error"], state["db_error"]) == ("SELECT 1", None, None)
    assert state["answer"] == "Stub answer: the query returned 1 rows."


def test_a_validation_failure_is_repaired_once_and_then_succeeds() -> None:
    state = run(validate_sql=fake("validate_sql", INVALID, VALID))

    assert visited(state) == [*TO_VALIDATION, *REPAIR, "execute", "answer"]
    assert state["attempts"] == 1
    assert state["answer"].startswith("Stub answer")


def test_a_database_error_is_repaired_and_validated_again() -> None:
    state = run(execute=fake("execute", DB_ERROR, DB_OK))

    assert visited(state) == [*TO_VALIDATION, "execute", *REPAIR, "execute", "answer"]
    assert (state["attempts"], state["db_error"]) == (1, None)
    assert state["answer"].startswith("Stub answer")


def test_repeated_validation_failures_stop_after_two_repairs() -> None:
    state = run(validate_sql=fake("validate_sql", INVALID))

    assert visited(state) == [*TO_VALIDATION, *REPAIR, *REPAIR, "answer"]
    assert state["attempts"] == MAX_REPAIRS == 2
    assert "execute" not in visited(state)  # invalid SQL never reaches the database
    assert state["answer"] == (
        "I could not answer this: the query still failed after 2 repairs "
        "(only SELECT queries are allowed)."
    )


def test_repeated_database_errors_stop_after_two_repairs() -> None:
    state = run(execute=fake("execute", DB_ERROR))

    assert visited(state) == [
        *TO_VALIDATION,
        "execute",
        *REPAIR,
        "execute",
        *REPAIR,
        "execute",
        "answer",
    ]
    assert state["attempts"] == 2
    assert 'column "nope" does not exist' in state["answer"]


def test_repairs_are_shared_between_validation_and_database_failures() -> None:
    state = run(
        validate_sql=fake("validate_sql", INVALID, VALID),
        execute=fake("execute", DB_ERROR),
    )

    # One repair for the invalid SQL, one for the first database error, then it gives up.
    assert visited(state) == [*TO_VALIDATION, *REPAIR, "execute", *REPAIR, "execute", "answer"]
    assert state["attempts"] == 2


def test_a_refused_question_ends_after_the_guardrail() -> None:
    refusal = {"refused": True, "refusal_reason": "That is investment advice."}

    state = run(guardrail=fake("guardrail", refusal))

    assert visited(state) == ["guardrail"]
    assert state["refusal_reason"] == "That is investment advice."
    assert "sql" not in state and "answer" not in state


def test_the_real_validation_node_stops_unsafe_sql_from_being_executed() -> None:
    state = run(generate_sql=fake("generate_sql", {"sql": "DELETE FROM companies"}))

    assert "execute" not in visited(state)
    assert state["validation_error"] == "only SELECT queries are allowed, not DELETE"
    assert state["attempts"] == 2


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"refused": True}, END),
        ({"refused": False}, "intent"),
        ({}, "intent"),
    ],
)
def test_route_after_guardrail(state: AgentState, expected: str) -> None:
    assert route_after_guardrail(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"validation_error": None}, "execute"),
        ({}, "execute"),
        ({"validation_error": "bad", "attempts": 0}, "repair_sql"),
        ({"validation_error": "bad"}, "repair_sql"),
        ({"validation_error": "bad", "attempts": 1}, "repair_sql"),
        ({"validation_error": "bad", "attempts": 2}, "answer"),
        ({"validation_error": "bad", "attempts": 3}, "answer"),
    ],
)
def test_route_after_validation(state: AgentState, expected: str) -> None:
    assert route_after_validation(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"db_error": None}, "answer"),
        ({}, "answer"),
        ({"db_error": "boom", "attempts": 0}, "repair_sql"),
        ({"db_error": "boom", "attempts": 1}, "repair_sql"),
        ({"db_error": "boom", "attempts": 2}, "answer"),
    ],
)
def test_route_after_execute(state: AgentState, expected: str) -> None:
    assert route_after_execute(state) == expected


def test_only_known_nodes_can_be_replaced() -> None:
    with pytest.raises(ValueError, match="no such nodes: summarise"):
        build_graph({"summarise": fake("summarise", {})})
    assert list(NODES) == [*TO_VALIDATION, "execute", "repair_sql", "answer"]


def test_the_saved_diagram_matches_the_graph() -> None:
    drawn = build_graph().get_graph().draw_mermaid().rstrip() + "\n"

    assert GRAPH_FILE.read_text() == drawn, (
        "docs/graph.mmd is stale; regenerate it from build_graph().get_graph().draw_mermaid()"
    )
