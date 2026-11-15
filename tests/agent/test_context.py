"""Tests for the text the SQL prompts show about companies, periods and examples."""

from app.agent.context import (
    NO_COMPANIES,
    NO_EXAMPLES,
    NO_PERIODS,
    describe_companies,
    describe_periods,
    format_examples,
)
from app.agent.periods import resolve_periods
from tests.agent.fixtures import CALENDAR


def periods(question: str, *tickers: str) -> str:
    return describe_periods(resolve_periods(question, list(tickers), CALENDAR)[0])


def test_companies_are_named_with_their_tickers() -> None:
    names = {"MSFT": "Microsoft", "GOOGL": "Alphabet"}

    assert describe_companies(["MSFT", "GOOGL"], names) == "MSFT (Microsoft), GOOGL (Alphabet)"
    assert describe_companies(["ZZZ"], names) == "ZZZ"
    assert describe_companies([], names) == NO_COMPANIES


def test_a_fiscal_year_is_shown_with_its_quarters_and_dates() -> None:
    assert periods("revenue in fiscal 2025", "MSFT") == (
        "- MSFT: fiscal 2025 Q1-Q4; period_end 2024-09-30 to 2025-06-30"
    )


def test_a_relative_phrase_says_what_it_means_and_which_quarter_it_is() -> None:
    assert periods("Apple last quarter", "AAPL") == (
        "- AAPL: the latest quarter in the data: fiscal 2026 Q4; period_end 2026-09-30"
    )
    assert periods("the last 6 quarters", "WMT") == (
        "- WMT: the latest 6 quarters in the data: fiscal 2026 Q1-Q4, fiscal 2027 Q1-Q2; "
        "period_end 2025-04-30 to 2026-07-31"
    )
    assert periods("debt since fiscal 2026", "MSFT").startswith("- MSFT: fiscal 2026 onwards: ")


def test_each_company_gets_its_own_line_and_an_assumed_year_is_flagged() -> None:
    text = periods("Google vs Microsoft 2023", "GOOGL", "MSFT")

    assert text.splitlines() == [
        "- GOOGL: fiscal 2023 Q1-Q4; period_end 2023-03-31 to 2023-12-31 "
        "(the question did not say fiscal or calendar; the fiscal year is assumed)",
        "- MSFT: fiscal 2023 Q1-Q4; period_end 2022-09-30 to 2023-06-30 "
        "(the question did not say fiscal or calendar; the fiscal year is assumed)",
    ]


def test_a_calendar_year_is_given_as_dates() -> None:
    assert periods("revenue in calendar 2025", "MSFT") == (
        "- MSFT: calendar year 2025, dates 2025-01-01 to 2025-12-31 "
        "(its quarters ending in that range: fiscal 2025 Q3-Q4, fiscal 2026 Q1-Q2)"
    )


def test_without_a_company_the_period_applies_to_each_company() -> None:
    assert periods("top 5 by revenue in fiscal 2025") == "- each company: fiscal 2025"
    assert periods("revenue in Q3 FY25") == "- each company: fiscal 2025 Q3"
    assert periods("total assets at the latest quarter") == (
        "- each company: the latest quarter in the data (its own newest rows, by period_end)"
    )
    assert periods("dividends in calendar 2025") == (
        "- each company: calendar year 2025 (2025-01-01 to 2025-12-31)"
    )


def test_no_periods_and_non_consecutive_quarters() -> None:
    assert describe_periods([]) == NO_PERIODS
    rows = [
        {
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "fiscal_quarter": q,
            "period_end": end,
            "basis": "fiscal",
        }
        for q, end in ((1, "2024-12-31"), (3, "2025-06-30"))
    ]
    assert (
        describe_periods(rows) == "- AAPL: fiscal 2025 Q1, Q3; period_end 2024-12-31 to 2025-06-30"
    )


def test_examples_are_shown_as_question_and_sql_pairs() -> None:
    examples = [{"question": "One?", "sql": "SELECT 1"}, {"question": "Two?", "sql": "SELECT 2"}]

    assert format_examples(examples) == "Q: One?\nSQL: SELECT 1\n\nQ: Two?\nSQL: SELECT 2"
    assert format_examples([]) == NO_EXAMPLES
