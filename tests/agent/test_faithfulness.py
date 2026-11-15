"""Tests for the faithfulness rules: reading numbers out of text and tracing them to rows."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.agent.faithfulness import (
    MAX_CLAIMS,
    MAX_DERIVED_VALUES,
    Claim,
    extract_numbers,
    unsupported_numbers,
)

D = Decimal


def claim(text: str, value: float, step: float, **flags: bool) -> Claim:
    return Claim(text=text, value=value, step=step, **flags)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1.2B", [claim("$1.2B", 1.2e9, 1e8)]),
        ("1.2 billion", [claim("1.2 billion", 1.2e9, 1e8)]),
        ("1.2 Billion USD", [claim("1.2 Billion", 1.2e9, 1e8)]),
        ("$3.4M", [claim("$3.4M", 3.4e6, 1e5)]),
        ("12bn", [claim("12bn", 12e9, 1e9)]),
        ("3k", [claim("3k", 3e3, 1e3)]),
        ("2 trillion", [claim("2 trillion", 2e12, 1e12)]),
        ("24.5%", [claim("24.5%", 24.5, 0.1, percent=True)]),
        ("5 percent", [claim("5 percent", 5.0, 1.0, percent=True)]),
        ("12.5 percentage points", [claim("12.5 percentage points", 12.5, 0.1, percent=True)]),
        ("1,234", [claim("1,234", 1234.0, 1.0)]),
        ("1,234,567.89 shares", [claim("1,234,567.89", 1234567.89, 0.01)]),
        (".5", [claim(".5", 0.5, 0.1)]),
        # negatives: a minus sign in either position, the typographic minus, accounting brackets
        ("-5", [claim("-5", 5.0, 1.0, negative=True)]),
        ("−5", [claim("−5", 5.0, 1.0, negative=True)]),
        ("-$1.2B", [claim("-$1.2B", 1.2e9, 1e8, negative=True)]),
        ("$-5", [claim("$-5", 5.0, 1.0, negative=True)]),
        ("($5)", [claim("($5)", 5.0, 1.0, negative=True)]),
        ("a loss of -3.5%", [claim("-3.5%", 3.5, 0.1, percent=True, negative=True)]),
        # punctuation and prose around a number are not part of it
        ("It was 1,234.", [claim("1,234", 1234.0, 1.0)]),
        ("It was $6.13, up.", [claim("$6.13", 6.13, 0.01)]),
        ("(2025)", [claim("2025", 2025.0, 1.0)]),
        ("**$1.2B**", [claim("$1.2B", 1.2e9, 1e8)]),
        # a hyphen between numbers is a range, not a minus
        ("2021-2025", [claim("2021", 2021.0, 1.0), claim("2025", 2025.0, 1.0)]),
        ("5-10%", [claim("5", 5.0, 1.0), claim("10%", 10.0, 1.0, percent=True)]),
        # an ISO date is a year, a month and a day
        (
            "2026-09-30",
            [claim("2026", 2026.0, 1.0), claim("09", 9.0, 1.0), claim("30", 30.0, 1.0)],
        ),
        # digits inside a word are labels, and a lower-case unit letter is not a scale
        ("Q3 and FY2025 and q1", []),
        ("5m", [claim("5", 5.0, 1.0)]),
        ("7 T", [claim("7", 7.0, 1.0)]),
        ("no numbers here", []),
        ("", []),
    ],
)
def test_extract_numbers(text: str, expected: list[Claim]) -> None:
    found = extract_numbers(text)

    assert [(c.text, c.percent, c.negative) for c in found] == [
        (c.text, c.percent, c.negative) for c in expected
    ]
    assert [c.value for c in found] == pytest.approx([c.value for c in expected])
    assert [c.step for c in found] == pytest.approx([c.step for c in expected])


def test_several_numbers_come_back_in_order() -> None:
    text = "Apple earned $416.2 billion in fiscal 2025, 6.4% more, or $7.46 per share."

    assert [c.text for c in extract_numbers(text)] == ["$416.2 billion", "2025", "6.4%", "$7.46"]


ROWS: list[tuple[Any, ...]] = [
    ("AAPL", 2025, D("416161000000.0000"), D("0.4621"), D("7.4600"), date(2025, 9, 27)),
    ("MSFT", 2025, D("281724000000.0000"), D("0.6882"), D("-1.2500"), date(2025, 6, 30)),
]


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue was 416,161,000,000 USD.",
        "Revenue was $416.2 billion.",  # rounded to the digit shown
        "Revenue was $416B.",
        "Revenue was 416,161 million USD.",
        "Revenue was $0.42 trillion.",
        "Diluted EPS was $7.46 per share.",
        "Gross margin was 46.21%.",  # a stored fraction written as a percentage
        "Gross margin was 46.2%, against 68.8% for Microsoft.",
        "Gross margin was 0.4621.",
        "EPS was -1.25 USD.",  # a negative written as one
        "The loss was $1.25 per share.",  # or as a loss, without the sign
        "Apple's revenue was $134.4 billion higher.",  # difference of two row values
        "Apple's revenue was 1.48 times Microsoft's.",  # ratio
        "Apple's revenue was 47.7% higher.",  # percentage change
        "Microsoft's revenue was 67.7% of Apple's.",  # ratio as a percentage
        "The margins differ by 22.61 percentage points.",  # difference, written as a percentage
        "In fiscal 2025, for the year ended 2025-09-27.",  # a column value and a date
        "The quarter ended on 27 September 2025.",
        "Both of the 2 companies reported.",  # the row count
        "No figures at all.",
        "",
    ],
)
def test_supported_answers(answer: str) -> None:
    assert unsupported_numbers(answer, ROWS) == []


@pytest.mark.parametrize(
    ("answer", "unsupported"),
    [
        ("Revenue was $500 billion.", ["$500 billion"]),
        ("Revenue was $416.9 billion.", ["$416.9 billion"]),  # outside the rounding shown
        ("Revenue was 416,161,000,001 USD.", ["416,161,000,001"]),
        ("Revenue was $416.2 million.", ["$416.2 million"]),  # the wrong scale
        ("Gross margin was 46.9%.", ["46.9%"]),
        ("EPS was -7.46 USD.", ["-7.46"]),  # a minus on a figure that is positive
        ("Revenue grew 12.3% to $416.2 billion in 2024.", ["12.3%", "2024"]),
        ("It rose 31% and 31% again.", ["31%"]),  # reported once
    ],
)
def test_unsupported_answers(answer: str, unsupported: list[str]) -> None:
    assert unsupported_numbers(answer, ROWS) == unsupported


def test_numbers_from_the_question_and_its_periods_may_be_repeated() -> None:
    answer = "Of the top 5 by revenue since fiscal 2021, 3M led with $416.2 billion."
    allowed = "Which were the top 5 companies since fiscal 2021?\nMMM (3M)"

    assert unsupported_numbers(answer, ROWS) == ["5", "2021", "3M"]
    assert unsupported_numbers(answer, ROWS, allowed) == []


def test_numbers_inside_text_cells_and_timestamps_count() -> None:
    rows = [("3M Company", "fiscal 2024 Q2", datetime(2024, 6, 30, 12, 0))]

    assert unsupported_numbers("3M Company, fiscal 2024, to 30 June (month 6).", rows) == []
    assert unsupported_numbers("3M Company in 2023.", rows) == ["2023"]


def test_date_parts_and_the_row_count_support_only_bare_numbers() -> None:
    rows = [(date(2025, 9, 27), 100.0), (date(2025, 6, 30), 50.0)]

    assert unsupported_numbers("The 2 quarters ended on 27 September and 30 June 2025.", rows) == []
    assert unsupported_numbers("It rose 9% to $27 and 2 million.", rows) == [
        "9%",
        "$27",
        "2 million",
    ]
    # The question's own numbers may be repeated in any form.
    assert unsupported_numbers("Growth above 9% was rare.", rows, "Who grew more than 9%?") == []


def test_values_that_are_not_figures_are_ignored() -> None:
    rows = [(True, None, float("nan"), float("inf"), "text", 7)]

    assert unsupported_numbers("It was 7.", rows) == []
    assert unsupported_numbers("It was 1.", rows) == []  # the row count, not the boolean
    assert unsupported_numbers("It was 8.", rows) == ["8"]


def test_no_rows_support_only_a_zero_count() -> None:
    assert unsupported_numbers("There were 0 rows and $5 of revenue.", []) == ["$5"]


def test_derived_values_are_not_tried_on_large_results() -> None:
    small = [(float(n),) for n in range(1000, 1000 + MAX_DERIVED_VALUES)]
    large = [*small, (5000.0,)]

    # 1199 - 1000 = 199, found by pairing two values of the small result only.
    assert unsupported_numbers("The gap was 199.", small) == []
    assert unsupported_numbers("The gap was 199.", large) == ["199"]
    assert unsupported_numbers("The top value was 5,000.", large) == []


def test_unrelated_cells_are_never_combined() -> None:
    rows = [("AAPL", 2025, 10.0, 4.0), ("MSFT", 2024, 7.0, 2.0)]

    assert unsupported_numbers("The gap was 3.", rows) == []  # 10 - 7, one column
    assert unsupported_numbers("It was 2.5 times.", rows) == []  # 10 / 4, one row
    assert unsupported_numbers("It was 40%.", rows) == []  # 4 / 10, one row
    assert unsupported_numbers("The gap was 6.", rows) == ["6"]  # 10 - 4 is across columns
    assert unsupported_numbers("It was 5 times.", rows) == ["5"]  # 10 / 2: other row and column
    assert unsupported_numbers("It was 2015.", rows) == ["2015"]  # 2025 - 10


def test_a_division_by_a_zero_value_is_skipped() -> None:
    assert unsupported_numbers("It was 3 times as much.", [(0,), (6,), (2,)]) == []
    assert unsupported_numbers("It was 7 times as much.", [(0,), (6,), (2,)]) == ["7"]


def test_a_very_long_answer_is_checked_in_bounded_time() -> None:
    """The answer is model output: thousands of invented numbers must not stall the run."""
    import time

    rows = [(float(n) * 1.37,) for n in range(1, MAX_DERIVED_VALUES + 1)]
    invented = [f"{n}.123%" for n in range(100_000, 105_000)]
    started = time.perf_counter()

    unsupported = unsupported_numbers("It was 1.37 and " + ", ".join(invented), rows)

    assert time.perf_counter() - started < 2.0
    assert unsupported == invented  # the first MAX_CLAIMS were examined, the rest are not trusted


def test_numbers_past_the_limit_are_reported_even_if_the_rows_hold_them() -> None:
    rows = [(float(n),) for n in range(1, MAX_CLAIMS + 3)]
    answer = ", ".join(f"{n}.0" for n in range(1, MAX_CLAIMS + 3))

    assert unsupported_numbers(answer, rows) == [f"{MAX_CLAIMS + 1}.0", f"{MAX_CLAIMS + 2}.0"]
