"""Tests for the answer node: a scripted model, fixture reference data, no database."""

from datetime import date
from decimal import Decimal
from typing import Any, get_type_hints

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from app import llm
from app.agent.nodes import answer as answer_module
from app.agent.nodes import generate_sql as generate_module
from app.agent.nodes.answer import (
    MAX_ANSWER_CHARS,
    MAX_PROMPT_ROWS,
    answer,
    format_rows,
    format_value,
)
from app.agent.state import AgentState
from app.db.reference import Company
from app.llm import LLMError, RateLimitedError
from tests.fakes import ScriptedChat, reply
from tests.test_llm import provider_error

D = Decimal
STATE_KEYS = set(get_type_hints(AgentState))
QUARTER_ENDS = ((1, "2024-09-30"), (2, "2024-12-31"), (3, "2025-03-31"), (4, "2025-06-30"))
PERIOD = {"ticker": "MSFT", "fiscal_year": 2025, "basis": "fiscal", "assumed": False}
STATE: dict[str, Any] = {
    "question": "What was Microsoft's total revenue in fiscal 2025?",
    "tickers": ["MSFT"],
    "periods": [
        PERIOD | {"fiscal_quarter": quarter, "period_end": end, "relative": None}
        for quarter, end in QUARTER_ENDS
    ],
    "sql": "SELECT 1",
    "columns": ["ticker", "total_revenue"],
    "rows": [("MSFT", D("281724000000.0000"))],
    "attempts": 0,
    "answer_attempts": 0,
}
GOOD = "Microsoft's total revenue in fiscal 2025 was 281.7 billion USD."


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ScriptedChat:
    """A scripted model behind the answer node, with fixture reference data (no database)."""
    fake = ScriptedChat(replies=[])

    def invoke(messages: list[BaseMessage]) -> AIMessage:
        return llm.invoke(messages, model=fake)

    monkeypatch.setattr(answer_module, "invoke", invoke)
    companies = (Company("MSFT", "Microsoft", "Technology", 6),)
    monkeypatch.setattr(generate_module.reference, "companies", lambda: companies)
    monkeypatch.setattr(
        generate_module.reference, "data_range", lambda: (date(2021, 1, 2), date(2026, 11, 13))
    )
    return fake


def test_the_answer_is_the_models_text_and_counts_the_attempt(model: ScriptedChat) -> None:
    model.replies = [reply(f"<think>281.7?</think>\n  {GOOD}\n")]

    update = answer(STATE)

    assert set(update) == {"answer", "answer_attempts", "trace"}
    assert set(update) <= STATE_KEYS
    assert (update["answer"], update["answer_attempts"]) == (GOOD, 1)
    assert update["trace"][0] | {"duration_ms": 0} == {
        "node": "answer",
        "duration_ms": 0,
        "note": "attempt 1",
    }


def test_the_prompt_shows_the_question_what_it_resolved_to_and_the_rows(
    model: ScriptedChat,
) -> None:
    model.replies = [reply(GOOD)]

    answer(STATE)

    (message,) = model.seen[0]
    prompt = message.content
    assert message.type == "human"  # no system role
    assert "Question: What was Microsoft's total revenue in fiscal 2025?" in prompt
    assert "Companies in this question: MSFT (Microsoft)" in prompt
    assert "- MSFT: fiscal 2025 Q1-Q4; period_end 2024-09-30 to 2025-06-30" in prompt
    assert "Query result (1 row):\nticker | total_revenue\nMSFT | 281724000000\n" in prompt
    assert "Every number you write must be a number in the result" in prompt
    assert "previous answer" not in prompt


def test_a_second_attempt_names_the_numbers_that_were_rejected(model: ScriptedChat) -> None:
    model.replies = [reply(GOOD)]
    state = STATE | {"answer_attempts": 1, "unsupported_numbers": ["$300 billion", "12%"]}

    update = answer(state)

    prompt = model.seen[0][0].content
    assert "these numbers are not in the result: $300 billion, 12%." in prompt
    assert update["answer_attempts"] == 2
    assert update["trace"][0]["note"] == "attempt 2, after unsupported: $300 billion, 12%"


def test_a_line_broken_reply_becomes_one_line(model: ScriptedChat) -> None:
    model.replies = [reply("Revenue was\n281.7 billion USD.\n\nThat is for fiscal 2025.")]

    assert answer(STATE)["answer"] == "Revenue was 281.7 billion USD. That is for fiscal 2025."


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"validation_error": "bad SQL", "attempts": 2},
            "I could not answer this: the query still failed after 2 repairs.",
        ),
        (
            {"db_error": 'relation "secrets" does not exist', "attempts": 1, "rows": [(1,)]},
            "I could not answer this: the query still failed after 1 repair.",
        ),
    ],
)
def test_a_failed_query_gets_a_fixed_sentence_without_a_model_call(
    model: ScriptedChat, state: AgentState, expected: str
) -> None:
    update = answer(state)

    error = state.get("validation_error") or state.get("db_error")
    assert set(update) == {"answer", "trace"}
    assert update["answer"] == expected  # the error is not shown to the user
    assert update["trace"][0]["note"] == f"gave up: {error}"  # it is kept in the trace
    assert model.seen == []


def test_an_overlong_reply_is_cut_before_it_is_checked(model: ScriptedChat) -> None:
    model.replies = [reply("Revenue was 281.7 billion USD. " * 100)]

    text = answer(STATE)["answer"]

    assert len(text) == MAX_ANSWER_CHARS and text.endswith("…")
    assert text.startswith("Revenue was 281.7 billion USD. Revenue")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            STATE | {"rows": []},
            "No data was found for MSFT (Microsoft) in the period asked about: "
            "the query returned no rows.",
        ),
        (
            STATE | {"rows": [], "periods": []},
            "No data was found for MSFT (Microsoft): the query returned no rows.",
        ),
        (
            {"question": "Which company lost money?", "rows": []},
            "No data was found for this question: the query returned no rows.",
        ),
        ({"question": "?"}, "No data was found for this question: the query returned no rows."),
    ],
)
def test_an_empty_result_gets_a_fixed_sentence_without_a_model_call(
    model: ScriptedChat, state: AgentState, expected: str
) -> None:
    update = answer(state)

    assert set(update) == {"answer", "trace"}
    assert update["answer"] == expected
    assert update["trace"][0]["note"] == "no rows"
    assert model.seen == []


@pytest.mark.parametrize(
    ("failure", "error"),
    [(provider_error(429), RateLimitedError), (provider_error(503), LLMError)],
)
def test_a_model_failure_is_raised(
    model: ScriptedChat, failure: Exception, error: type[Exception]
) -> None:
    model.replies = [failure]

    with pytest.raises(error):
        answer(STATE)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, "NULL"),
        (D("281724000000.0000"), "281724000000"),
        (D("0.4621"), "0.4621"),
        (D("7.4600"), "7.46"),
        (D("100"), "100"),
        (D("1E+3"), "1000"),
        (D("-1.50"), "-1.5"),
        (12, "12"),
        (0.5, "0.5"),
        (date(2025, 6, 30), "2025-06-30"),
        ("Microsoft", "Microsoft"),
    ],
)
def test_format_value(value: Any, text: str) -> None:
    assert format_value(value) == text


def test_format_rows_shows_a_limited_number_of_rows_and_says_so() -> None:
    table, note = format_rows(["ticker", "close"], [("AAPL", D("1.50")), ("MSFT", None)])
    assert (table, note) == ("ticker | close\nAAPL | 1.5\nMSFT | NULL", "2 rows")

    many = [(n,) for n in range(MAX_PROMPT_ROWS + 5)]
    table, note = format_rows(["n"], many)
    assert len(table.splitlines()) == MAX_PROMPT_ROWS + 1
    assert note == f"the first {MAX_PROMPT_ROWS} of {MAX_PROMPT_ROWS + 5} rows"
