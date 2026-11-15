"""Turns time phrases in a question into explicit fiscal periods, without a model.

Everything is relative to the data, never to today's date: "last quarter" is the latest fiscal
quarter a company has in the database. Periods are resolved per company, because fiscal years
differ (Apple's ends in September, Microsoft's in June, Walmart's in January).

What is understood
------------------
- "last / latest / most recent quarter"            the newest quarter in the data
- "last 8 quarters", "past four quarters", "TTM"   the newest N quarters
- "Q3 2025", "third quarter of fiscal 2025"        one quarter of a fiscal year
- "FY2024", "fiscal 2024", "fiscal year 2024"      the quarters of that fiscal year
- "2023" on its own                                the same, marked `assumed`: the question did
                                                   not say fiscal or calendar
- "calendar 2025"                                  quarters ending inside that calendar year
- "2022 to 2025", "between 2022 and 2025"          every fiscal year in the range
- "since fiscal 2022"                              from that year to the newest in the data
- "last fiscal year", "latest fiscal year"         the newest fiscal year that is complete
- "YTD", "year to date", "this fiscal year"        the quarters of the newest fiscal year

A full date ("30 June 2026") is not a fiscal period and is left for the SQL to handle.

Each result is a dict: `ticker, fiscal_year, fiscal_quarter, period_start, period_end` (ISO
dates), `basis` ("fiscal" or "calendar"), `assumed`, and `relative` (the phrase's meaning, such
as "latest_quarter", when the period was found relative to the data). With no company named,
the result describes the request without dates, for the SQL to apply to whichever companies
it selects.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.db.reference import Calendar, Quarter

NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "twelve": 12,
}  # fmt: skip
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
}
MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_FISCAL = r"(?:fy|fiscal(?:\s+year)?)"
_YEAR = r"(?:20\d{2})"
_RECENT = r"(?:last|latest|most\s+recent|previous|newest)"

LAST_N_QUARTERS = re.compile(
    rf"\b(?:{_RECENT}|past|trailing)\s+(\d{{1,2}}|{'|'.join(NUMBER_WORDS)})\s+(?:fiscal\s+)?quarters\b"
)
TRAILING_YEAR = re.compile(r"\b(?:ttm|trailing\s+twelve\s+months|trailing\s+12\s+months)\b")
LATEST_QUARTER = re.compile(rf"\b{_RECENT}\s+(?:reported\s+)?(?:fiscal\s+)?quarter\b(?!s)")
LATEST_YEAR = re.compile(rf"\b{_RECENT}\s+(?:completed\s+|full\s+)?fiscal\s+year\b")
YEAR_TO_DATE = re.compile(
    r"\b(?:ytd|year[\s-]to[\s-]date|so\s+far\s+this\s+(?:fiscal\s+)?year|this\s+fiscal\s+year)\b"
)
QUARTER_OF_YEAR = re.compile(
    rf"\b(?:q([1-4])|({'|'.join(ORDINALS)})\s+quarter)\s*(?:of\s+|in\s+)?({_FISCAL}\s*)?'?(\d{{4}}|\d{{2}})\b"
)
CALENDAR_YEAR = re.compile(rf"\bcalendar(?:\s+year)?\s+({_YEAR})\b")
YEAR_RANGE = re.compile(
    rf"\b(?:(fiscal\s+years?\s+)?(from|between)\s+)?({_FISCAL}\s*)?({_YEAR})\s*(to|through|until|and|-|–)\s*"
    rf"({_FISCAL}\s*)?({_YEAR})\b"
)
SINCE_YEAR = re.compile(rf"\bsince\s+({_FISCAL}\s*)?({_YEAR})\b")
FISCAL_YEAR = re.compile(rf"\b{_FISCAL}\s*'?(\d{{4}}|\d{{2}})\b")
BARE_YEAR = re.compile(rf"\b({_YEAR})\b")
FULL_DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS})|(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?),?\s+{_YEAR}\b"
    rf"|\b{_YEAR}-\d{{2}}-\d{{2}}\b"
)


@dataclass(frozen=True)
class Request:
    """One time phrase, before it is applied to a company's calendar."""

    kind: str  # latest_quarter, last_quarters, quarter, fiscal_year, calendar_year,
    #            since_year, latest_year, year_to_date
    year: int | None = None
    quarter: int | None = None
    count: int | None = None
    assumed: bool = False

    @property
    def relative(self) -> str | None:
        if self.kind == "last_quarters":
            return f"last_{self.count}_quarters"
        if self.kind == "since_year":
            return f"since_fiscal_{self.year}"
        return self.kind if self.kind in ("latest_quarter", "latest_year", "year_to_date") else None


def _year(text: str) -> int:
    return int(text) if len(text) == 4 else 2000 + int(text)


def parse_requests(question: str) -> list[Request]:
    """Find the time phrases in a question. Text a phrase used is not read again."""
    text = question.lower()
    requests: list[Request] = []

    def take(pattern: re.Pattern[str]) -> list[re.Match[str]]:
        nonlocal text
        matches = list(pattern.finditer(text))
        for match in matches:  # blank the phrase out, keeping every position
            text = text[: match.start()] + " " * (match.end() - match.start()) + text[match.end() :]
        return matches

    take(FULL_DATE)
    for match in take(LAST_N_QUARTERS):
        word = match.group(1)
        requests.append(Request("last_quarters", count=NUMBER_WORDS.get(word) or int(word)))
    if take(TRAILING_YEAR):
        requests.append(Request("last_quarters", count=4))
    if take(LATEST_QUARTER):
        requests.append(Request("latest_quarter"))
    if take(LATEST_YEAR):
        requests.append(Request("latest_year"))
    if take(YEAR_TO_DATE):
        requests.append(Request("year_to_date"))
    for match in take(QUARTER_OF_YEAR):
        quarter = int(match.group(1)) if match.group(1) else ORDINALS[match.group(2)]
        requests.append(
            Request("quarter", _year(match.group(4)), quarter, assumed=match.group(3) is None)
        )
    for match in take(CALENDAR_YEAR):
        requests.append(Request("calendar_year", int(match.group(1))))
    for match in YEAR_RANGE.finditer(text):
        fiscal_lead, opener, fiscal_a, first, joiner, fiscal_b, last = match.groups()
        if joiner == "and" and opener != "between":
            continue  # "fiscal 2024 and 2025" is two years, not a range; read below
        text = text[: match.start()] + " " * (match.end() - match.start()) + text[match.end() :]
        low, high = sorted((int(first), int(last)))
        assumed = fiscal_lead is None and fiscal_a is None and fiscal_b is None
        requests += [Request("fiscal_year", year, assumed=assumed) for year in range(low, high + 1)]
    for match in take(SINCE_YEAR):
        requests.append(Request("since_year", int(match.group(2)), assumed=match.group(1) is None))
    for match in take(FISCAL_YEAR):
        requests.append(Request("fiscal_year", _year(match.group(1))))
    for match in take(BARE_YEAR):
        requests.append(Request("fiscal_year", int(match.group(1)), assumed=True))
    # The same period can be named twice ("fiscal 2025 ... during 2025"); the explicit one wins.
    unique: dict[tuple[Any, ...], Request] = {}
    for request in sorted(requests, key=lambda r: r.assumed):
        unique.setdefault((request.kind, request.year, request.quarter, request.count), request)
    return [r for r in requests if unique.get((r.kind, r.year, r.quarter, r.count)) is r]


def _select(request: Request, quarters: tuple[Quarter, ...]) -> list[Quarter]:
    """The quarters of one company that a request covers."""
    if not quarters:
        return []
    newest_year = quarters[-1].fiscal_year
    if request.kind == "latest_quarter":
        return [quarters[-1]]
    if request.kind == "last_quarters":
        # A slice from -0 would be the whole list, so "last 0 quarters" must be handled first.
        return list(quarters[-request.count :]) if request.count else []
    if request.kind == "quarter":
        return [
            q
            for q in quarters
            if (q.fiscal_year, q.fiscal_quarter) == (request.year, request.quarter)
        ]
    if request.kind == "fiscal_year":
        return [q for q in quarters if q.fiscal_year == request.year]
    if request.kind == "calendar_year":
        return [q for q in quarters if q.period_end.year == request.year]
    if request.kind == "since_year":
        return [q for q in quarters if q.fiscal_year >= (request.year or 0)]
    if request.kind == "year_to_date":
        return [q for q in quarters if q.fiscal_year == newest_year]
    if request.kind == "latest_year":
        complete = [q.fiscal_year for q in quarters if q.fiscal_quarter == 4]
        return [q for q in quarters if complete and q.fiscal_year == complete[-1]]
    return []


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _row(ticker: str | None, request: Request, quarter: Quarter | None) -> dict[str, Any]:
    calendar_span = request.kind == "calendar_year" and quarter is None
    return {
        "ticker": ticker,
        "fiscal_year": quarter.fiscal_year if quarter else _neutral_year(request),
        "fiscal_quarter": quarter.fiscal_quarter if quarter else request.quarter,
        "period_start": _iso(quarter.period_start)
        if quarter
        else (f"{request.year}-01-01" if calendar_span else None),
        "period_end": _iso(quarter.period_end)
        if quarter
        else (f"{request.year}-12-31" if calendar_span else None),
        "basis": "calendar" if request.kind == "calendar_year" else "fiscal",
        "assumed": request.assumed,
        "relative": request.relative,
    }


def _neutral_year(request: Request) -> int | None:
    return request.year if request.kind in ("fiscal_year", "quarter", "since_year") else None


def resolve_periods(
    question: str, tickers: list[str], calendar: Calendar
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve a question's time phrases for the given companies.

    Returns the period rows and notes about anything that could not be found in the data.
    With no tickers, each phrase gives one row with `ticker` None and no dates looked up.
    """
    requests = parse_requests(question)
    if not tickers:
        return [_row(None, request, None) for request in requests], []
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for ticker in tickers:
        for request in requests:
            quarters = _select(request, calendar.get(ticker, ()))
            if not quarters:
                notes.append(f"no data for {ticker} {_label(request)}")
            rows += [_row(ticker, request, quarter) for quarter in quarters]
    unique = {(r["ticker"], r["fiscal_year"], r["fiscal_quarter"], r["basis"]): r for r in rows}
    # Oldest first within each company, whatever order the phrases came in.
    ordered = sorted(unique.values(), key=lambda r: (tickers.index(r["ticker"]), r["period_end"]))
    return ordered, notes


def _label(request: Request) -> str:
    if request.kind == "quarter":
        return f"fiscal {request.year} Q{request.quarter}"
    if request.kind == "calendar_year":
        return f"calendar {request.year}"
    if request.year:
        return f"fiscal {request.year}"
    return request.kind.replace("_", " ")
