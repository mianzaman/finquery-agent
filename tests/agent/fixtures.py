"""Fixture companies and fiscal calendars for the agent tests, so they need no database."""

from datetime import date, timedelta

from app.db.reference import Calendar, Company, Quarter

COMPANIES = (
    Company("AAPL", "Apple", "Technology", 9),
    Company("MSFT", "Microsoft", "Technology", 6),
    Company("GOOGL", "Alphabet", "Communication Services", 12),
    Company("WMT", "Walmart", "Consumer Defensive", 1),
    Company("META", "Meta Platforms", "Communication Services", 12),
    Company("V", "Visa", "Financial Services", 9),
    Company("SO", "Southern Company", "Utilities", 12),
    Company("D", "Dominion Energy", "Utilities", 12),
    Company("GE", "GE Aerospace", "Industrials", 12),
    Company("GS", "Goldman Sachs", "Financial Services", 12),
    Company("JPM", "JPMorgan Chase", "Financial Services", 12),
    Company("HD", "Home Depot", "Consumer Cyclical", 1),
)
ALIASES = {
    "GOOGL": ["Google", "Alphabet Inc"],
    "META": ["Facebook", "Meta"],
    "JPM": ["JPMorgan", "JP Morgan", "Chase"],
    "GE": ["General Electric"],
    "SO": ["Southern"],
}


def _month_end(year: int, month: int) -> date:
    first_of_next = date(year + month // 12, month % 12 + 1, 1)
    return first_of_next - timedelta(days=1)


def quarters(year_end_month: int, first_year: int, last: tuple[int, int]) -> tuple[Quarter, ...]:
    """Quarters ending on month ends, from Q1 of `first_year` to `last` (year, quarter).

    A fiscal year is named after the calendar year it ends in, as the real companies do
    (Walmart's year ending January 2026 is fiscal 2026).
    """
    result: list[Quarter] = []
    for fiscal_year in range(first_year, last[0] + 1):
        for quarter in range(1, 5):
            if (fiscal_year, quarter) > last:
                break
            months_before_year_end = (4 - quarter) * 3
            month_index = fiscal_year * 12 + (year_end_month - 1) - months_before_year_end
            end = _month_end(month_index // 12, month_index % 12 + 1)
            start_index = month_index - 2
            start = date(start_index // 12, start_index % 12 + 1, 1)
            result.append(Quarter(fiscal_year, quarter, start, end))
    return tuple(result)


# The data "ends" in autumn 2026, like the real database.
CALENDAR: Calendar = {
    "AAPL": quarters(9, 2022, (2026, 4)),  # latest: quarter ended 2026-09-30
    "MSFT": quarters(6, 2022, (2027, 1)),  # latest: fiscal 2027 Q1, ended 2026-09-30
    "GOOGL": quarters(12, 2021, (2026, 3)),
    "WMT": quarters(1, 2022, (2027, 2)),  # latest: fiscal 2027 Q2, ended 2026-07-31
    "META": quarters(12, 2021, (2026, 3)),
    "V": quarters(9, 2022, (2026, 4)),
    "JPM": quarters(12, 2021, (2026, 3)),
}
