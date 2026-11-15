"""Route tests for the agent graph: nodes are replaced with fakes, so no LLM or database is used."""

from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END

from app.agent.graph import (
    MAX_ANSWER_ATTEMPTS,
    MAX_REPAIRS,
    NODES,
    build_graph,
    initial_state,
    route_after_answer,
    route_after_check_answer,
    route_after_entities,
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
ANSWERED = ["answer", "check_answer"]


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


ALLOWED = {"refused": False, "refusal_reason": None, "intent": "trend", "company_mentions": []}
RESOLVED = {"tickers": ["AAPL"], "periods": []}
RETRIEVED = {"schema_context": "companies (\n  ticker text,\n)", "examples": []}


def fake_repair(state: AgentState) -> dict[str, Any]:
    """Like the real repair node, without a model: same SQL back, one more attempt."""
    return {
        "sql": state.get("sql"),
        "validation_error": None,
        "db_error": None,
        "attempts": state.get("attempts", 0) + 1,
        "trace": trace_event("repair_sql", 0.0, "fake"),
    }


def fake_answer(state: AgentState) -> dict[str, Any]:
    """The real node's fixed sentences (they need no model); otherwise an answer from the rows."""
    if state.get("validation_error") or state.get("db_error") or not state.get("rows"):
        return NODES["answer"](state)
    return {
        "answer": "The value was 1.",
        "answer_attempts": state.get("answer_attempts", 0) + 1,
        "trace": trace_event("answer", 0.0, "fake"),
    }


FAITHFUL = {"unsupported_numbers": [], "faithfulness_failed": False}
UNFAITHFUL = {"unsupported_numbers": ["$9 billion"], "faithfulness_failed": False}


def run(**overrides: Any) -> AgentState:
    """Run the graph with every model-backed and database-backed node replaced by a fake."""
    front = {
        "guardrail": fake("guardrail", ALLOWED),
        "entities": fake("entities", RESOLVED),
        "retrieve_schema": fake("retrieve_schema", RETRIEVED),
        "generate_sql": fake("generate_sql", {"sql": "SELECT 1"}),
        "validate_sql": fake("validate_sql", VALID),
        "execute": fake("execute", DB_OK),
        "repair_sql": fake_repair,
        "answer": fake_answer,
        "check_answer": fake("check_answer", FAITHFUL),
    }
    return build_graph(front | overrides).invoke(initial_state("What was Apple's revenue?"))


def test_happy_path_visits_every_node_in_order() -> None:
    state = run()

    assert visited(state) == [*TO_VALIDATION, "execute", *ANSWERED]
    assert state["question"] == "What was Apple's revenue?"
    assert (state["refused"], state["attempts"]) == (False, 0)
    assert (state["intent"], state["tickers"]) == ("trend", ["AAPL"])
    assert (state["sql"], state["validation_error"], state["db_error"]) == ("SELECT 1", None, None)
    assert (state["answer"], state["answer_attempts"]) == ("The value was 1.", 1)
    assert (state["unsupported_numbers"], state["faithfulness_failed"]) == ([], False)


def test_a_validation_failure_is_repaired_once_and_then_succeeds() -> None:
    state = run(validate_sql=fake("validate_sql", INVALID, VALID))

    assert visited(state) == [*TO_VALIDATION, *REPAIR, "execute", *ANSWERED]
    assert state["attempts"] == 1
    assert state["answer"] == "The value was 1."


def test_a_database_error_is_repaired_and_validated_again() -> None:
    state = run(execute=fake("execute", DB_ERROR, DB_OK))

    assert visited(state) == [*TO_VALIDATION, "execute", *REPAIR, "execute", *ANSWERED]
    assert (state["attempts"], state["db_error"]) == (1, None)
    assert state["answer"] == "The value was 1."


def test_repeated_validation_failures_stop_after_two_repairs() -> None:
    state = run(validate_sql=fake("validate_sql", INVALID))

    assert visited(state) == [*TO_VALIDATION, *REPAIR, *REPAIR, "answer"]
    assert state["attempts"] == MAX_REPAIRS == 2
    assert "execute" not in visited(state)  # invalid SQL never reaches the database
    assert state["answer"] == "I could not answer this: the query still failed after 2 repairs."
    assert state["trace"][-1]["note"] == "gave up: only SELECT queries are allowed"


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
    assert 'column "nope" does not exist' not in state["answer"]
    assert state["db_error"] == 'column "nope" does not exist'


def test_repairs_are_shared_between_validation_and_database_failures() -> None:
    state = run(
        validate_sql=fake("validate_sql", INVALID, VALID),
        execute=fake("execute", DB_ERROR),
    )

    # One repair for the invalid SQL, one for the first database error, then it gives up.
    assert visited(state) == [*TO_VALIDATION, *REPAIR, "execute", *REPAIR, "execute", "answer"]
    assert state["attempts"] == 2


def test_an_unfaithful_answer_is_written_again_once() -> None:
    state = run(check_answer=fake("check_answer", UNFAITHFUL, FAITHFUL))

    assert visited(state) == [*TO_VALIDATION, "execute", *ANSWERED, *ANSWERED]
    assert state["answer_attempts"] == 2
    assert (state["answer"], state["faithfulness_failed"]) == ("The value was 1.", False)


def test_two_unfaithful_answers_end_in_the_neutral_sentence() -> None:
    state = run(check_answer=fake("check_answer", UNFAITHFUL))

    assert visited(state) == [*TO_VALIDATION, "execute", *ANSWERED, *ANSWERED, "fallback_answer"]
    assert state["answer_attempts"] == MAX_ANSWER_ATTEMPTS == 2
    assert state["answer"] == "The query returned 1 row; the figures are in the table."
    assert state["faithfulness_failed"] is True
    assert state["rows"] == [(1,)]  # the rows are still returned


@pytest.fixture
def reference_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture companies behind the real answer and check nodes, so they need no database."""
    from datetime import date

    from app.agent.nodes import generate_sql as generate_module
    from tests.agent.fixtures import COMPANIES

    monkeypatch.setattr(generate_module.reference, "companies", lambda: COMPANIES)
    monkeypatch.setattr(
        generate_module.reference, "data_range", lambda: (date(2021, 1, 2), date(2026, 11, 13))
    )


def test_the_fixed_sentences_are_not_checked(reference_data: None) -> None:
    empty = run(execute=fake("execute", DB_OK | {"rows": []}))
    failed = run(validate_sql=fake("validate_sql", INVALID))

    for state in (empty, failed):
        assert visited(state)[-1] == "answer"
        assert "unsupported_numbers" not in state and "faithfulness_failed" not in state
    assert empty["answer"] == "No data was found for AAPL (Apple): the query returned no rows."


def test_the_real_check_sends_an_invented_number_back_to_the_model(
    reference_data: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real answer, check and fallback nodes with a scripted model."""
    from langchain_core.messages import AIMessage, BaseMessage

    from app import llm
    from app.agent.nodes import answer as answer_module
    from tests.fakes import ScriptedChat, reply

    rows = {"db_error": None, "columns": ["revenue"], "rows": [(416161000000,)]}
    real = {name: NODES[name] for name in ("answer", "check_answer", "fallback_answer")}

    def run_with(*answers: str) -> tuple[AgentState, ScriptedChat]:
        model = ScriptedChat(replies=[reply(text) for text in answers])

        def invoke(messages: list[BaseMessage]) -> AIMessage:
            return llm.invoke(messages, model=model)

        monkeypatch.setattr(answer_module, "invoke", invoke)
        return run(execute=fake("execute", rows), **real), model

    state, model = run_with("Revenue was $500 billion.", "Revenue was $416.2 billion.")
    assert visited(state) == [*TO_VALIDATION, "execute", *ANSWERED, *ANSWERED]
    assert (state["answer"], state["faithfulness_failed"]) == ("Revenue was $416.2 billion.", False)
    assert "these numbers are not in the result: $500 billion." in model.seen[1][0].content

    state, model = run_with("Revenue was $500 billion.", "Revenue was $501 billion.")
    assert visited(state)[-3:] == [*ANSWERED, "fallback_answer"]
    assert state["answer"] == "The query returned 1 row; the figures are in the table."
    assert (state["faithfulness_failed"], state["unsupported_numbers"]) == (True, ["$501 billion"])
    assert len(model.seen) == 2  # never a third answer


def test_a_refused_question_ends_after_the_guardrail() -> None:
    refusal = {"refused": True, "refusal_reason": "That is investment advice."}

    state = run(guardrail=fake("guardrail", refusal))

    assert visited(state) == ["guardrail"]
    assert state["refusal_reason"] == "That is investment advice."
    assert "sql" not in state and "answer" not in state


def test_an_unknown_company_ends_after_entities() -> None:
    unknown = {
        "refused": True,
        "refusal_reason": 'I could not find "Samsung" among the companies I have data for.',
        "tickers": [],
        "periods": [],
    }

    state = run(entities=fake("entities", unknown))

    assert visited(state) == ["guardrail", "intent", "entities"]
    assert state["refused"] is True and "Samsung" in state["refusal_reason"]
    assert "sql" not in state and "answer" not in state


def test_run_agent_uses_the_real_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # The pre-check refuses this without a model call, so the real graph can run in a test.
    state = run_agent("DROP TABLE companies")

    assert visited(state) == ["guardrail"]
    assert state["refused"] is True


def test_the_real_validation_node_stops_unsafe_sql_from_being_executed() -> None:
    state = run(
        generate_sql=fake("generate_sql", {"sql": "DELETE FROM companies"}),
        validate_sql=NODES["validate_sql"],
    )

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
        ({"refused": True}, END),
        ({"refused": False}, "retrieve_schema"),
        ({}, "retrieve_schema"),
    ],
)
def test_route_after_entities(state: AgentState, expected: str) -> None:
    assert route_after_entities(state) == expected


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


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"rows": [(1,)]}, "check_answer"),
        ({"rows": [(1,)], "validation_error": None, "db_error": None}, "check_answer"),
        ({"rows": []}, END),
        ({}, END),
        ({"rows": [(1,)], "db_error": "boom"}, END),
        ({"rows": [(1,)], "validation_error": "bad"}, END),
    ],
)
def test_route_after_answer(state: AgentState, expected: str) -> None:
    assert route_after_answer(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"unsupported_numbers": []}, END),
        ({}, END),
        ({"unsupported_numbers": [], "answer_attempts": 1}, END),
        ({"unsupported_numbers": ["5%"], "answer_attempts": 1}, "answer"),
        ({"unsupported_numbers": ["5%"]}, "answer"),
        ({"unsupported_numbers": ["5%"], "answer_attempts": 2}, "fallback_answer"),
        ({"unsupported_numbers": ["5%"], "answer_attempts": 3}, "fallback_answer"),
    ],
)
def test_route_after_check_answer(state: AgentState, expected: str) -> None:
    assert route_after_check_answer(state) == expected


def test_only_known_nodes_can_be_replaced() -> None:
    with pytest.raises(ValueError, match="no such nodes: summarise"):
        build_graph({"summarise": fake("summarise", {})})
    assert list(NODES) == [
        *TO_VALIDATION,
        "execute",
        "repair_sql",
        "answer",
        "check_answer",
        "fallback_answer",
    ]


def test_the_saved_diagram_matches_the_graph() -> None:
    drawn = build_graph().get_graph().draw_mermaid().rstrip() + "\n"

    assert GRAPH_FILE.read_text() == drawn, (
        "docs/graph.mmd is stale; regenerate it from build_graph().get_graph().draw_mermaid()"
    )


def test_broken_sql_from_the_model_is_repaired_in_one_attempt(
    db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real SQL nodes with a scripted model: a bad column, then a corrected query."""
    from pydantic import BaseModel

    from app import llm
    from app.agent.nodes import generate_sql as generate_module
    from app.agent.nodes import repair_sql as repair_module
    from tests.fakes import ScriptedChat, reply

    broken = "SELECT revenu FROM income_statements WHERE ticker = 'AAPL'"
    fixed = "SELECT ticker, name FROM companies WHERE ticker = 'AAPL'"
    model = ScriptedChat(
        replies=[
            reply(f'{{"sql": "{broken}", "tables_used": ["income_statements"]}}'),
            reply(f'{{"sql": "{fixed}", "tables_used": ["companies"]}}'),
        ]
    )

    def structured(schema: type[BaseModel], messages: list) -> BaseModel:
        return llm.structured(schema, messages, model=model, native=False)

    monkeypatch.setattr(generate_module, "structured", structured)
    monkeypatch.setattr(repair_module, "structured", structured)
    real = {name: NODES[name] for name in ("generate_sql", "validate_sql", "execute", "repair_sql")}

    state = run(**real)

    assert visited(state) == [*TO_VALIDATION, *REPAIR, "execute", *ANSWERED]
    assert state["attempts"] == 1
    assert state["sql"] == f"{fixed} LIMIT 500"  # what ran is the validated, row-limited text
    assert (state["columns"], state["rows"]) == (["ticker", "name"], [("AAPL", "Apple")])
    assert (state["validation_error"], state["db_error"]) == (None, None)
    # The repair prompt showed the model its query and the database's exact complaint.
    repair_prompt = model.seen[1][0].content
    assert broken in repair_prompt and 'column "revenu" does not exist' in repair_prompt
    assert len(model.seen) == 2
