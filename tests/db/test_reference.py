"""Tests for the reference data loader, against the local database."""

import pytest

from app.db import reference

pytestmark = pytest.mark.usefixtures("db")


def test_companies_have_tickers_names_and_fiscal_year_ends() -> None:
    companies = reference.companies()

    assert len(companies) >= 1
    assert [c.ticker for c in companies] == sorted(c.ticker for c in companies)
    assert all(c.name for c in companies)
    assert all(
        c.fiscal_year_end_month is None or 1 <= c.fiscal_year_end_month <= 12 for c in companies
    )
    assert reference.companies() is companies  # read once per process


def test_the_fiscal_calendar_is_oldest_first_for_every_company() -> None:
    calendar = reference.fiscal_calendar()

    assert set(calendar) <= {c.ticker for c in reference.companies()}
    for quarters in calendar.values():
        ends = [q.period_end for q in quarters]
        assert ends == sorted(ends)
        assert all(1 <= q.fiscal_quarter <= 4 for q in quarters)
        assert len({(q.fiscal_year, q.fiscal_quarter) for q in quarters}) == len(quarters)


def test_the_data_range_spans_the_financials_and_the_prices() -> None:
    first, last = reference.data_range()
    calendar = reference.fiscal_calendar()

    if not calendar:
        assert first is None
        return
    assert first == min(q.period_end for quarters in calendar.values() for q in quarters)
    assert last >= max(q.period_end for quarters in calendar.values() for q in quarters)
