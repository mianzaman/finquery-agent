"""Reference data the agent needs before it writes SQL: the companies and their fiscal calendars.

Read once per process, as the read-only role, through `run_select`. Nodes use these functions
instead of querying the database themselves.
"""

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from app.db.session import run_select

COMPANIES_QUERY = """
    SELECT ticker, name, sector, fiscal_year_end_month
    FROM companies
    ORDER BY ticker
"""
QUARTERS_QUERY = """
    SELECT ticker, fiscal_year, fiscal_quarter, period_start, period_end
    FROM income_statements
    ORDER BY ticker, period_end
"""


DATA_RANGE_QUERY = """
    SELECT
        (SELECT min(period_end) FROM income_statements),
        GREATEST(
            (SELECT max(period_end) FROM income_statements),
            (SELECT max(trade_date) FROM daily_prices)
        )
"""


@dataclass(frozen=True)
class Company:
    """One row of `companies`."""

    ticker: str
    name: str
    sector: str | None
    fiscal_year_end_month: int | None


@dataclass(frozen=True)
class Quarter:
    """One fiscal quarter a company has data for."""

    fiscal_year: int
    fiscal_quarter: int
    period_start: date | None
    period_end: date


Calendar = dict[str, tuple[Quarter, ...]]


@lru_cache
def companies() -> tuple[Company, ...]:
    """Every company in the dataset, by ticker."""
    _, rows, _ = run_select(COMPANIES_QUERY)
    return tuple(Company(*row) for row in rows)


@lru_cache
def fiscal_calendar() -> Calendar:
    """Each company's fiscal quarters in the data, oldest first."""
    _, rows, truncated = run_select(QUARTERS_QUERY, max_rows=50_000)
    if truncated:
        raise RuntimeError("the fiscal calendar has more rows than the loader reads")
    calendar: dict[str, list[Quarter]] = {}
    for ticker, *quarter in rows:
        calendar.setdefault(ticker, []).append(Quarter(*quarter))
    return {ticker: tuple(quarters) for ticker, quarters in calendar.items()}


@lru_cache
def data_range() -> tuple[date | None, date | None]:
    """The first and last dates the data covers (None, None for an empty database).

    A model's sense of "today" is its training date, so without this it calls recent periods
    "the future" and refuses them.
    """
    _, rows, _ = run_select(DATA_RANGE_QUERY)
    first, last = rows[0]
    return first, last
