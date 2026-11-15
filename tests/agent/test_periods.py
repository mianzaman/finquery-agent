"""Tests for the period rules. Pure functions on fixture calendars: no database, no model."""

from typing import Any

import pytest

from app.agent.periods import Request, parse_requests, resolve_periods
from tests.agent.fixtures import CALENDAR, quarters


def R(kind: str, year: int | None = None, quarter: int | None = None, **more: Any) -> Request:
    return Request(kind, year, quarter, **more)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Apple's net income in its most recent quarter", [R("latest_quarter")]),
        ("cash at the end of its latest reported quarter", [R("latest_quarter")]),
        ("revenue last quarter", [R("latest_quarter")]),
        ("revenue over its last 8 quarters", [R("last_quarters", count=8)]),
        ("the past four quarters", [R("last_quarters", count=4)]),
        ("TTM revenue", [R("last_quarters", count=4)]),
        ("revenue in FY2024", [R("fiscal_year", 2024)]),
        ("revenue in fiscal 2024", [R("fiscal_year", 2024)]),
        ("revenue in fiscal year 2024", [R("fiscal_year", 2024)]),
        ("revenue in FY24 and FY25", [R("fiscal_year", 2024), R("fiscal_year", 2025)]),
        ("Google vs Microsoft 2023", [R("fiscal_year", 2023, assumed=True)]),
        ("revenue in calendar 2025", [R("calendar_year", 2025)]),
        ("revenue in calendar year 2025", [R("calendar_year", 2025)]),
        ("revenue in Q3 2025", [R("quarter", 2025, 3, assumed=True)]),
        ("revenue in Q3 FY25", [R("quarter", 2025, 3)]),
        ("the third quarter of fiscal 2025", [R("quarter", 2025, 3)]),
        (
            "free cash flow in each fiscal year from 2022 to 2024",
            [R("fiscal_year", 2022), R("fiscal_year", 2023), R("fiscal_year", 2024)],
        ),
        (
            "revenue between 2023 and 2024",
            [R("fiscal_year", 2023, assumed=True), R("fiscal_year", 2024, assumed=True)],
        ),
        ("debt since fiscal 2022", [R("since_year", 2022)]),
        ("debt since 2022", [R("since_year", 2022, assumed=True)]),
        ("revenue YTD", [R("year_to_date")]),
        ("revenue year to date", [R("year_to_date")]),
        ("revenue in the latest fiscal year", [R("latest_year")]),
        # The explicit mention wins over a second, bare mention of the same year.
        ("grew in fiscal 2025 but the share price fell during 2025", [R("fiscal_year", 2025)]),
        # A full date is not a fiscal period.
        ("closing share price on 30 June 2026", []),
        ("closing price on March 3, 2025 and on 2024-05-01", []),
        ("Should I buy Nvidia stock right now?", []),
        ("Which 5 companies have the most employees?", []),  # "5" is not a year
        (
            "the last 8 quarters and the last quarter",
            [R("last_quarters", count=8), R("latest_quarter")],
        ),
    ],
)
def test_parse_requests(question: str, expected: list[Request]) -> None:
    assert parse_requests(question) == expected


def keys(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(r["ticker"], r["fiscal_year"], r["fiscal_quarter"], r["period_end"]) for r in rows]


def test_last_quarter_is_the_companys_own_latest_fiscal_quarter() -> None:
    rows, notes = resolve_periods("Apple last quarter", ["AAPL"], CALENDAR)

    # Apple's fiscal year ends in September, so its latest quarter is fiscal Q4, not calendar Q3.
    assert keys(rows) == [("AAPL", 2026, 4, "2026-09-30")]
    assert rows[0] == {
        "ticker": "AAPL",
        "fiscal_year": 2026,
        "fiscal_quarter": 4,
        "period_start": "2026-07-01",
        "period_end": "2026-09-30",
        "basis": "fiscal",
        "assumed": False,
        "relative": "latest_quarter",
    }
    assert notes == []


def test_the_same_phrase_gives_each_company_its_own_period() -> None:
    rows, _ = resolve_periods("revenue last quarter", ["AAPL", "MSFT", "WMT"], CALENDAR)

    assert keys(rows) == [
        ("AAPL", 2026, 4, "2026-09-30"),
        ("MSFT", 2027, 1, "2026-09-30"),
        ("WMT", 2027, 2, "2026-07-31"),
    ]


def test_a_bare_year_is_each_companys_fiscal_year_and_marked_assumed() -> None:
    rows, _ = resolve_periods("Google vs Microsoft 2023", ["GOOGL", "MSFT"], CALENDAR)

    assert keys(rows) == [
        ("GOOGL", 2023, 1, "2023-03-31"),
        ("GOOGL", 2023, 2, "2023-06-30"),
        ("GOOGL", 2023, 3, "2023-09-30"),
        ("GOOGL", 2023, 4, "2023-12-31"),
        # Microsoft's fiscal 2023 ran from July 2022 to June 2023.
        ("MSFT", 2023, 1, "2022-09-30"),
        ("MSFT", 2023, 2, "2022-12-31"),
        ("MSFT", 2023, 3, "2023-03-31"),
        ("MSFT", 2023, 4, "2023-06-30"),
    ]
    assert all(row["assumed"] and row["basis"] == "fiscal" for row in rows)


def test_a_calendar_year_takes_the_quarters_that_end_inside_it() -> None:
    rows, _ = resolve_periods("revenue in calendar 2025", ["MSFT", "WMT"], CALENDAR)

    assert keys(rows) == [
        ("MSFT", 2025, 3, "2025-03-31"),
        ("MSFT", 2025, 4, "2025-06-30"),
        ("MSFT", 2026, 1, "2025-09-30"),
        ("MSFT", 2026, 2, "2025-12-31"),
        ("WMT", 2025, 4, "2025-01-31"),
        ("WMT", 2026, 1, "2025-04-30"),
        ("WMT", 2026, 2, "2025-07-31"),
        ("WMT", 2026, 3, "2025-10-31"),
    ]
    assert {row["basis"] for row in rows} == {"calendar"}


def fy(*years: int) -> list[tuple[int, int]]:
    """All four quarters of each fiscal year."""
    return [(year, quarter) for year in years for quarter in (1, 2, 3, 4)]


@pytest.mark.parametrize(
    ("question", "ticker", "expected"),
    [
        ("the past 8 quarters", "AAPL", fy(2025, 2026)),
        ("revenue YTD", "WMT", [(2027, 1), (2027, 2)]),
        ("revenue YTD", "MSFT", [(2027, 1)]),
        ("Q3 FY25", "MSFT", [(2025, 3)]),
        # The newest fiscal year that is complete: Microsoft's fiscal 2027 has only begun.
        ("the latest fiscal year", "MSFT", fy(2026)),
        ("the latest fiscal year", "AAPL", fy(2026)),
        ("since fiscal 2026", "WMT", [*fy(2026), (2027, 1), (2027, 2)]),
        ("FY24 and FY25", "V", fy(2024, 2025)),
    ],
)
def test_phrases_resolve_against_the_data(
    question: str, ticker: str, expected: list[tuple[int, int]]
) -> None:
    rows, notes = resolve_periods(question, [ticker], CALENDAR)

    assert [(r["fiscal_year"], r["fiscal_quarter"]) for r in rows] == expected
    assert notes == []


def test_a_period_outside_the_data_is_noted_not_invented() -> None:
    rows, notes = resolve_periods("revenue in fiscal 2019 and Q1 2030", ["AAPL"], CALENDAR)

    assert rows == []
    assert notes == ["no data for AAPL fiscal 2030 Q1", "no data for AAPL fiscal 2019"]


def test_a_company_without_a_calendar_is_noted() -> None:
    rows, notes = resolve_periods("revenue last quarter", ["ZZZZ"], CALENDAR)

    assert (rows, notes) == ([], ["no data for ZZZZ latest quarter"])


def test_overlapping_phrases_do_not_repeat_a_quarter() -> None:
    rows, _ = resolve_periods("fiscal 2026 and the last quarter", ["AAPL"], CALENDAR)

    assert [(r["fiscal_year"], r["fiscal_quarter"]) for r in rows] == [
        (2026, q) for q in range(1, 5)
    ]


def test_without_a_company_the_request_is_described_but_not_dated() -> None:
    rows, notes = resolve_periods(
        "Which companies had the most revenue last quarter?", [], CALENDAR
    )
    assert notes == []
    assert rows == [
        {
            "ticker": None,
            "fiscal_year": None,
            "fiscal_quarter": None,
            "period_start": None,
            "period_end": None,
            "basis": "fiscal",
            "assumed": False,
            "relative": "latest_quarter",
        }
    ]

    fiscal, _ = resolve_periods("top 5 by revenue in fiscal 2025", [], CALENDAR)
    assert (fiscal[0]["fiscal_year"], fiscal[0]["period_end"], fiscal[0]["relative"]) == (
        2025,
        None,
        None,
    )
    calendar, _ = resolve_periods("dividends in calendar 2025", [], CALENDAR)
    assert (calendar[0]["period_start"], calendar[0]["period_end"]) == ("2025-01-01", "2025-12-31")
    assert calendar[0]["basis"] == "calendar"


def test_last_zero_quarters_is_nothing_not_everything() -> None:
    assert resolve_periods("the last 0 quarters", ["AAPL"], CALENDAR)[0] == []
    assert len(resolve_periods("the last 99 quarters", ["AAPL"], CALENDAR)[0]) == 20


def test_no_time_phrase_gives_no_periods() -> None:
    assert resolve_periods("What sector is Apple in?", ["AAPL"], CALENDAR) == ([], [])


def test_the_fixture_calendar_matches_real_fiscal_years() -> None:
    apple = quarters(9, 2025, (2025, 4))
    assert [str(q.period_end) for q in apple] == [
        "2024-12-31",
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
    ]
    walmart = quarters(1, 2026, (2026, 1))
    assert (str(walmart[0].period_start), str(walmart[0].period_end)) == (
        "2025-02-01",
        "2025-04-30",
    )
