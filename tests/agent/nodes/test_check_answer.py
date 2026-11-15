"""Tests for check_answer and fallback_answer: no model, fixture reference data, no database."""

from datetime import date
from decimal import Decimal
from typing import Any, get_type_hints

import pytest

from app.agent.nodes import generate_sql as generate_module
from app.agent.nodes.check_answer import allowed_text, check_answer
from app.agent.nodes.fallback_answer import fallback_answer
from app.agent.state import AgentState
from app.db.reference import Company

STATE_KEYS = set(get_type_hints(AgentState))
STATE: dict[str, Any] = {
    "question": "What were the top 3 quarters of Microsoft's revenue in fiscal 2025?",
    "tickers": ["MSFT"],
    "periods": [
        {
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "fiscal_quarter": 4,
            "basis": "fiscal",
            "assumed": False,
            "period_end": "2025-06-30",
            "relative": None,
        }
    ],
    "columns": ["fiscal_quarter", "revenue"],
    "rows": [(4, Decimal("76441000000.0000")), (2, Decimal("69632000000.0000"))],
}


@pytest.fixture(autouse=True)
def reference_data(monkeypatch: pytest.MonkeyPatch) -> None:
    companies = (Company("MSFT", "Microsoft", "Technology", 6),)
    monkeypatch.setattr(generate_module.reference, "companies", lambda: companies)
    monkeypatch.setattr(
        generate_module.reference, "data_range", lambda: (date(2021, 1, 2), date(2026, 11, 13))
    )


def test_a_faithful_answer_leaves_an_empty_list() -> None:
    text = "Of the top 3, fiscal 2025 Q4 was highest at $76.4 billion, then Q2 at $69.6 billion."

    update = check_answer(STATE | {"answer": text})

    assert set(update) == {"unsupported_numbers", "faithfulness_failed", "trace"}
    assert set(update) <= STATE_KEYS
    assert (update["unsupported_numbers"], update["faithfulness_failed"]) == ([], False)
    assert update["trace"][0]["node"] == "check_answer"
    assert update["trace"][0]["note"] == "every number is supported"


def test_an_invented_number_is_listed() -> None:
    text = "Revenue peaked at $76.4 billion in 2025, up 18% from $64.7 billion."

    update = check_answer(STATE | {"answer": text})

    assert update["unsupported_numbers"] == ["18%", "$64.7 billion"]
    assert update["trace"][0]["note"] == "unsupported: 18%, $64.7 billion"


def test_the_allowed_text_is_the_question_and_what_it_resolved_to() -> None:
    text = allowed_text(STATE)

    assert "top 3 quarters" in text
    assert "MSFT (Microsoft)" in text
    assert "- MSFT: fiscal 2025 Q4; period_end 2025-06-30" in text


def test_a_state_without_an_answer_or_rows_has_nothing_unsupported() -> None:
    assert check_answer({"question": "?"})["unsupported_numbers"] == []


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([(1,), (2,), (3,)], "The query returned 3 rows; the figures are in the table."),
        ([(1,)], "The query returned 1 row; the figures are in the table."),
    ],
)
def test_the_fallback_says_only_how_many_rows_there_are(rows: list[tuple], expected: str) -> None:
    update = fallback_answer({"rows": rows, "answer": "Revenue was $999 billion."})

    assert set(update) == {"answer", "faithfulness_failed", "trace"}
    assert set(update) <= STATE_KEYS
    assert (update["answer"], update["faithfulness_failed"]) == (expected, True)
    assert update["trace"][0]["node"] == "fallback_answer"
