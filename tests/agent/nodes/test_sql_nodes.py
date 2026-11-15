"""Tests for generate_sql, repair_sql and execute: a scripted model; execute uses the database."""

from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from app import llm
from app.agent.nodes import generate_sql as generate_module
from app.agent.nodes import repair_sql as repair_module
from app.agent.nodes.execute import execute
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.repair_sql import repair_sql
from app.db.reference import Company
from app.llm import LLMError, RateLimitedError
from tests.fakes import ScriptedChat, reply
from tests.test_llm import provider_error

SCHEMA_CONTEXT = (
    "-- income_statements: Quarterly income statement.\n"
    "income_statements (\n  ticker text,\n  revenue numeric,\n)"
)
EXAMPLE = {
    "question": "What was Alphabet's revenue in fiscal 2024?",
    "sql": "SELECT SUM(revenue) FROM income_statements WHERE ticker = 'GOOGL'",
}
QUARTER_ENDS = ((1, "2024-09-30"), (2, "2024-12-31"), (3, "2025-03-31"), (4, "2025-06-30"))
PERIOD = {"ticker": "MSFT", "fiscal_year": 2025, "basis": "fiscal", "assumed": False}
STATE: dict[str, Any] = {
    "question": "What was Microsoft's total revenue in fiscal 2025?",
    "schema_context": SCHEMA_CONTEXT,
    "examples": [EXAMPLE],
    "tickers": ["MSFT"],
    "periods": [
        PERIOD | {"fiscal_quarter": quarter, "period_end": end, "relative": None}
        for quarter, end in QUARTER_ENDS
    ],
    "attempts": 0,
}
GOOD = "SELECT SUM(revenue) FROM income_statements WHERE ticker = 'MSFT' AND fiscal_year = 2025"


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ScriptedChat:
    """A scripted model behind both SQL nodes, with fixture reference data (no database)."""
    fake = ScriptedChat(replies=[])

    def structured(schema: type[BaseModel], messages: list) -> BaseModel:
        return llm.structured(schema, messages, model=fake, native=False)

    companies = (Company("MSFT", "Microsoft", "Technology", 6),)
    for module in (generate_module, repair_module):
        monkeypatch.setattr(module, "structured", structured)
    monkeypatch.setattr(generate_module.reference, "companies", lambda: companies)
    monkeypatch.setattr(
        generate_module.reference, "data_range", lambda: (date(2021, 1, 2), date(2026, 11, 13))
    )
    return fake


def answer(sql: str, tables: str = "income_statements") -> Any:
    return reply(f'{{"sql": "{sql}", "tables_used": ["{tables}"]}}')


# --- generate_sql -----------------------------------------------------------------------------


def test_generate_returns_the_models_query(model: ScriptedChat) -> None:
    model.replies = [answer(f"  {GOOD}  ")]

    update = generate_sql(STATE)

    assert set(update) == {"sql", "trace"}
    assert update["sql"] == GOOD
    assert update["trace"][0] | {"duration_ms": 0} == {
        "node": "generate_sql",
        "duration_ms": 0,
        "note": "tables: income_statements",
    }


def test_the_generation_prompt_holds_everything_the_earlier_nodes_found(
    model: ScriptedChat,
) -> None:
    model.replies = [answer(GOOD)]

    generate_sql(STATE)

    (message,) = model.seen[0][:1]
    prompt = message.content
    assert message.type == "human"  # no system role
    assert "Question: What was Microsoft's total revenue in fiscal 2025?" in prompt
    assert "income_statements (\n  ticker text,\n  revenue numeric,\n)" in prompt
    assert "Q: What was Alphabet's revenue in fiscal 2024?\nSQL: SELECT SUM(revenue)" in prompt
    assert "Companies in this question: MSFT (Microsoft)" in prompt
    assert "- MSFT: fiscal 2025 Q1-Q4; period_end 2024-09-30 to 2025-06-30" in prompt
    assert "The data runs from January 2021 to 13 November 2026." in prompt
    assert "never CURRENT_DATE" in prompt and "Use only the tables and columns listed" in prompt


def test_generation_works_with_nothing_resolved_or_retrieved(model: ScriptedChat) -> None:
    model.replies = [answer("SELECT 1", "")]

    update = generate_sql({"question": "Which 5 companies had the highest revenue?"})

    prompt = model.seen[0][0].content
    assert update["sql"] == "SELECT 1"
    assert "Companies in this question: none named" in prompt
    assert "- none resolved; take the period from the question's own wording" in prompt
    assert "Similar questions with correct queries:\n(none)" in prompt


def test_a_reply_without_a_query_gives_empty_sql_for_validation_to_reject(
    model: ScriptedChat,
) -> None:
    model.replies = [reply('{"tables_used": []}')]

    update = generate_sql(STATE)

    assert update["sql"] == ""
    assert update["trace"][0]["note"] == "tables: none named"


@pytest.mark.parametrize(
    ("failure", "error"),
    [(provider_error(429), RateLimitedError), (provider_error(503), LLMError)],
)
def test_a_model_failure_in_generation_is_raised(
    model: ScriptedChat, failure: Exception, error: type[Exception]
) -> None:
    model.replies = [failure]

    with pytest.raises(error):
        generate_sql(STATE)


# --- repair_sql -------------------------------------------------------------------------------


def test_repair_returns_a_new_query_counts_the_attempt_and_clears_the_errors(
    model: ScriptedChat,
) -> None:
    model.replies = [answer(GOOD)]
    failed = STATE | {
        "sql": "SELECT revenu FROM income_statements",
        "validation_error": "bad",
        "db_error": None,
    }

    update = repair_sql(failed)

    assert set(update) == {"sql", "validation_error", "db_error", "attempts", "trace"}
    assert (update["sql"], update["attempts"]) == (GOOD, 1)
    assert (update["validation_error"], update["db_error"]) == (None, None)
    assert update["trace"][0]["note"] == "repair 1 after: bad"
    assert repair_sql_attempts(model, failed | {"attempts": 1}) == 2


def repair_sql_attempts(model: ScriptedChat, state: dict[str, Any]) -> int:
    model.replies = [answer(GOOD)]
    return repair_sql(state)["attempts"]


PLAN_ERROR = 'the database could not plan the query: column "revenu" does not exist'


@pytest.mark.parametrize(
    ("errors", "shown"),
    [
        ({"validation_error": PLAN_ERROR}, PLAN_ERROR),
        ({"db_error": "canceling statement due to statement timeout"}, "statement timeout"),
        # When both are set, the validation error is the newer one.
        (
            {"validation_error": "the function pg_sleep is not allowed", "db_error": "older"},
            "the function pg_sleep is not allowed",
        ),
        ({}, "the query failed"),
    ],
)
def test_the_repair_prompt_shows_the_failed_query_and_the_exact_error(
    model: ScriptedChat, errors: dict[str, str], shown: str
) -> None:
    model.replies = [answer(GOOD)]

    repair_sql(STATE | {"sql": "SELECT revenu FROM income_statements"} | errors)

    prompt = model.seen[0][0].content
    assert "The query that failed:\nSELECT revenu FROM income_statements" in prompt
    assert shown in prompt.split("The error:\n")[1].split("\n\n")[0]
    assert "Question: What was Microsoft's total revenue in fiscal 2025?" in prompt
    assert "income_statements (\n  ticker text,\n  revenue numeric,\n)" in prompt
    assert "Companies in this question: MSFT (Microsoft)" in prompt


def test_repair_without_any_query_says_so(model: ScriptedChat) -> None:
    model.replies = [answer(GOOD)]

    repair_sql({"question": "q", "validation_error": "no SQL statement was given"})

    assert "The query that failed:\n(no query was written)" in model.seen[0][0].content


def test_a_rate_limit_in_repair_is_raised(model: ScriptedChat) -> None:
    model.replies = [provider_error(429)]

    with pytest.raises(RateLimitedError):
        repair_sql(STATE | {"sql": "SELECT 1", "db_error": "boom"})


# --- execute ----------------------------------------------------------------------------------


def test_execute_runs_the_query_and_stores_columns_and_rows(db: None) -> None:
    sql = "SELECT n AS number, n * 2 AS doubled FROM generate_series(1, 3) AS n ORDER BY n"

    update = execute({"sql": sql})

    assert set(update) == {"columns", "rows", "db_error", "trace"}
    assert (update["columns"], update["rows"]) == (["number", "doubled"], [(1, 2), (2, 4), (3, 6)])
    assert update["db_error"] is None
    assert update["trace"][0]["note"] == "3 rows"


def test_execute_never_returns_more_than_500_rows(db: None) -> None:
    update = execute({"sql": "SELECT n FROM generate_series(1, 2000) AS n"})

    assert len(update["rows"]) == 500


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("SELECT nope FROM companies", 'column "nope" does not exist'),
        ("SELECT 1 / 0", "division by zero"),
        ("DELETE FROM companies", "read-only transaction"),
        ("SELECT 1; SELECT 2", "multiple commands"),
    ],
)
def test_a_database_error_becomes_db_error_and_no_rows(db: None, sql: str, error: str) -> None:
    update = execute({"sql": sql})

    assert set(update) == {"db_error", "trace"}
    assert error in update["db_error"]
    assert "\n" not in update["db_error"] and "localhost" not in update["db_error"]


def test_a_slow_query_is_cancelled_and_reported(db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.db.session.READER_STATEMENT_TIMEOUT_MS", 200)

    update = execute({"sql": "SELECT count(*) FROM generate_series(1, 100000000)"})

    assert "statement timeout" in update["db_error"]


@pytest.mark.parametrize("state", [{}, {"sql": None}, {"sql": ""}])
def test_execute_without_a_query_reports_it(state: dict[str, Any]) -> None:
    assert execute(state)["db_error"] == "there is no query to run"
