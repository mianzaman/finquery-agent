"""Loads quarterly financials from the SEC EDGAR XBRL API.

Run with `uv run python -m ingestion.load_sec [--tickers AAPL,MSFT] [--refresh]`.

How filed facts become quarterly rows
-------------------------------------
A company-facts file lists every reported number with its period dates and the filing it came
from. A filing repeats earlier periods as comparatives, so the `fy`/`fp` on a fact describe the
filing, not the fact. The loader therefore works from dates:

1. Fiscal calendar. The latest period end in each 10-K is a fiscal year end, labelled with that
   filing's `fy` (the company's own name for the year). The latest period end in each 10-Q is a
   quarter end; its quarter number is its distance from the previous fiscal year end. Q4 ends on
   the fiscal year end.
2. Income and cash-flow items cover a span of time. A 10-Q may report the quarter alone, the
   year to date, or both, and the 10-K reports only the full year. For each quarter the loader
   uses the three-month figure when one was filed; otherwise it subtracts the previous quarter's
   year-to-date figure from this quarter's. Q4 is therefore the full year minus the nine-month
   figure. Diluted EPS is derived the same way, which is the usual approximation: it can differ
   by a cent from a figure computed with Q4's own share count.
3. Balance sheet items are point-in-time: the value dated on the quarter end.
4. Every value is stored as first filed. A later filing often restates an earlier period, but
   only the periods it happens to show: after a stock split or a spin-off some quarters of a
   year are restated and others are not, and "latest wins" leaves the series on two bases.
   As first filed, each quarter matches the 10-Q or 10-K that originally reported it.
5. A quarter derived by subtraction needs both figures on the same basis. When the filing that
   supplies this year's year-to-date figure reports last year's figure for the same span
   differently from how it was first filed, that filing has restated history (a spin-off moved
   a business to discontinued operations, or a split changed the share count), while the
   previous quarter's figure is still on the old basis. If the previous quarter's filing
   restated its own comparative alike, both are on the new basis and the subtraction stands.
   Otherwise the quarter is left NULL for that column and listed under `warnings` in the
   report. Figures filed directly for the quarter are kept.
6. gross_profit, total_liabilities and (for oil and gas producers) capex are calculated when
   the company does not file them (see `derive_columns`); each row lists the columns that were
   calculated in `derived_columns`.
7. A company's facts can be spread over more than one SEC registrant (a predecessor after a
   reorganisation, or the co-registrant of a combined filing); `related_ciks.yaml` names them
   and `merge_related` adds the filings the main file lacks.
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import yaml
from sqlalchemy import Connection, Engine, MetaData, Table, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from app.config import LOG_FORMAT

ROOT = Path(__file__).resolve().parent.parent
COMPANIES_CSV = Path(__file__).parent / "companies.csv"
MAPPING_YAML = Path(__file__).parent / "xbrl_mapping.yaml"
RELATED_YAML = Path(__file__).parent / "related_ciks.yaml"
RAW_DIR = ROOT / "data" / "raw" / "sec"
REPORT_PATH = ROOT / "data" / "ingestion_report.json"

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CACHE_MAX_AGE = timedelta(days=7)
MIN_REQUEST_INTERVAL = 0.2  # seconds, so at most 5 requests per second
MAX_ATTEMPTS = 4
FISCAL_YEARS = 5

CALENDAR_FORMS = {"10-K": "annual", "10-Q": "quarterly"}
VALUE_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
UNITS = ("USD", "USD/shares")
QUARTER_DAYS = 91.25
START_TOLERANCE_DAYS = 10
RESTATEMENT_TOLERANCE = Decimal("0.01")  # a restated figure differs by more than 1%
SAME_BASIS_TOLERANCE = Decimal("0.10")  # two filings restated alike if within 10%

KEYS = {
    "income_statements": ["ticker", "fiscal_year", "fiscal_quarter"],
    "balance_sheets": ["ticker", "fiscal_year", "fiscal_quarter"],
    "cash_flows": ["ticker", "fiscal_year", "fiscal_quarter"],
    "companies": ["ticker"],
    "daily_prices": ["ticker", "trade_date"],
    "dividends": ["ticker", "ex_date"],
    "stock_splits": ["ticker", "split_date"],
}

logger = logging.getLogger(__name__)

Mapping = dict[str, Any]
Row = dict[str, Any]


class SecError(Exception):
    """SEC data could not be fetched or understood."""


@dataclass(frozen=True)
class Period:
    """One fiscal quarter of one company."""

    fiscal_year: int
    fiscal_quarter: int
    start: date | None
    end: date
    year_start: date
    filed: date | None = None  # when the 10-Q or 10-K that first reported the quarter was filed


@dataclass(frozen=True)
class TagFacts:
    """The usable facts of one XBRL tag. `durations` and `instants` hold the first filed value."""

    durations: dict[date, dict[date, Decimal]]  # end -> start -> value
    instants: dict[date, Decimal]  # end -> value
    source: dict[tuple[date, date], str] = field(default_factory=dict)  # span -> first filing
    # filing -> span -> the value that filing gave, including restated comparatives
    as_filed: dict[str, dict[tuple[date, date], Decimal]] = field(default_factory=dict)


@dataclass
class CompanyData:
    """Everything derived from one company-facts file."""

    fiscal_year_end_month: int
    rows: dict[str, list[Row]]
    missing_columns: list[str]
    warnings: list[str]


# --- Reading inputs ---------------------------------------------------------------------------


def read_companies(path: Path = COMPANIES_CSV) -> list[tuple[str, str]]:
    """Return `(ticker, name)` pairs from the owner's CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if not {"ticker", "name"} <= set(reader.fieldnames):
            raise SecError(f"{path.name} needs the columns ticker,name")
        return [(r["ticker"].strip().upper(), r["name"].strip()) for r in reader if r["ticker"]]


def read_mapping(path: Path = MAPPING_YAML) -> Mapping:
    """Return `{table: {column: [tags in order]}}`, plus the `derived_inputs` tag lists."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_related(path: Path = RELATED_YAML) -> dict[str, dict[str, list[int]]]:
    """Return `{ticker: {"predecessors": [cik], "co_registrants": [cik]}}`."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def cik_by_ticker(tickers_json: dict[str, Any]) -> dict[str, int]:
    """Index SEC's company tickers file by ticker. SEC writes share classes as `BRK-B`."""
    return {entry["ticker"].upper(): int(entry["cik_str"]) for entry in tickers_json.values()}


def resolve_cik(ticker: str, index: dict[str, int]) -> int:
    """Find a ticker's CIK, accepting `BRK.B` for SEC's `BRK-B`."""
    cik = index.get(ticker) or index.get(ticker.replace(".", "-"))
    if cik is None:
        raise SecError(f"{ticker} is not in SEC's company tickers file")
    return cik


# --- HTTP and cache ---------------------------------------------------------------------------


class SecClient:
    """A throttled HTTP client for sec.gov that retries rate limits and server errors."""

    def __init__(
        self,
        user_agent: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent.strip():
            raise SecError("SEC_USER_AGENT is empty; set it in .env to your name and email")
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = MIN_REQUEST_INTERVAL - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get_text(self, url: str) -> str:
        """GET a URL, retrying 429, 5xx and connection errors with exponential backoff."""
        problem = ""
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                self._sleep(2.0 ** (attempt - 1))
            self._throttle()
            try:
                response = self._client.get(url, headers=self._headers)
            except httpx.TransportError as exc:
                problem = type(exc).__name__
                continue
            if response.status_code == 429 or response.status_code >= 500:
                problem = f"HTTP {response.status_code}"
                continue
            if response.status_code != 200:
                raise SecError(f"HTTP {response.status_code} for {url}")
            return response.text
        raise SecError(f"gave up on {url} after {MAX_ATTEMPTS} attempts ({problem})")


def is_fresh(path: Path, now: datetime, max_age: timedelta = CACHE_MAX_AGE) -> bool:
    """True when a cache file exists and is younger than `max_age` (7 days for SEC files)."""
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return now - modified < max_age


def fetch_json(client: SecClient, url: str, cache_path: Path, refresh: bool) -> Any:
    """Return parsed JSON, from the cache when it is fresh, else downloaded and cached."""
    if refresh or not is_fresh(cache_path, datetime.now(tz=UTC)):
        text = client.get_text(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    else:
        text = cache_path.read_text(encoding="utf-8")
    try:
        # Decimal keeps reported numbers exact; floats would drift in subtraction.
        return json.loads(text, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError as exc:
        raise SecError(f"{cache_path.name} is not valid JSON") from exc


# --- Facts to quarters ------------------------------------------------------------------------


def _raw_facts(facts: dict[str, Any], tag: str) -> tuple[str, list[dict[str, Any]]]:
    units = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
    for unit in UNITS:
        if unit in units:
            return unit, units[unit]
    return "", []


def index_tag(facts: dict[str, Any], tag: str) -> TagFacts:
    """Collect a tag's 10-K/10-Q facts by period, keeping the first filed value (rule 4)."""
    indexed = TagFacts(durations={}, instants={})
    usable = [f for f in _raw_facts(facts, tag)[1] if f.get("form") in VALUE_FORMS]
    for fact in sorted(usable, key=lambda f: f.get("filed", "")):
        end = date.fromisoformat(fact["end"])
        value = Decimal(fact["val"])
        if "start" not in fact:
            indexed.instants.setdefault(end, value)
            continue
        start = date.fromisoformat(fact["start"])
        accn = fact.get("accn", "")
        indexed.as_filed.setdefault(accn, {})[(start, end)] = value
        if start not in indexed.durations.setdefault(end, {}):
            indexed.durations[end][start] = value
            indexed.source[(start, end)] = accn
    return indexed


def _filing_period_ends(facts: dict[str, Any], tags: list[str]) -> dict[str, dict[str, Any]]:
    """For each 10-K and 10-Q: the latest period end among the mapped tags, `fy` and filing day."""
    filings: dict[str, dict[str, Any]] = {}
    for tag in tags:
        for fact in _raw_facts(facts, tag)[1]:
            kind = CALENDAR_FORMS.get(fact.get("form", ""))
            if kind is None or "accn" not in fact:
                continue
            end = date.fromisoformat(fact["end"])
            filing = filings.setdefault(
                fact["accn"], {"kind": kind, "end": end, "fy": None, "filed": None}
            )
            filing["end"] = max(filing["end"], end)
            if fact.get("filed"):
                filing["filed"] = date.fromisoformat(fact["filed"])
            if fact.get("fy") is not None:
                filing["fy"] = int(fact["fy"])
    return filings


def fiscal_year_labels(year_ends: dict[date, int | None]) -> dict[date, int]:
    """Name each fiscal year, counting on from the first with the label most 10-Ks agree on.

    The label is the company's own (`fy` on its 10-K), but SEC has it wrong on the odd filing
    (Honeywell's 10-K for 2021 says 2020), so one filing is not trusted on its own.
    """
    ordered = sorted(year_ends)
    position = {end: round((end - ordered[0]).days / 365.25) for end in ordered}
    votes = Counter(year_ends[end] - position[end] for end in ordered if year_ends[end])
    if not votes:
        raise SecError("no 10-K filings with usable facts")
    first_label = votes.most_common(1)[0][0]
    return {end: first_label + position[end] for end in ordered}


def build_calendar(facts: dict[str, Any], tags: list[str]) -> list[Period]:
    """Work out the company's fiscal quarters from its filings (rule 1 in the module docstring)."""
    filings = _filing_period_ends(facts, tags).values()
    labels = fiscal_year_labels({f["end"]: f["fy"] for f in filings if f["kind"] == "annual"})
    quarter_ends = sorted({f["end"] for f in filings if f["kind"] == "quarterly"})
    # The first report for a period end is the one whose values are stored (rule 4).
    first_filed: dict[date, date] = {}
    for filing in filings:
        if filing["filed"] is not None:
            earliest = min(filing["filed"], first_filed.get(filing["end"], filing["filed"]))
            first_filed[filing["end"]] = earliest

    # Each fiscal year as (label, day after the previous year end, year end or None if open).
    ordered = sorted(labels)
    years: list[tuple[int, date, date | None]] = []
    previous = ordered[0] - timedelta(days=365)
    for end in ordered:
        years.append((labels[end], previous + timedelta(days=1), end))
        previous = end
    years.append((labels[ordered[-1]] + 1, previous + timedelta(days=1), None))

    periods: list[Period] = []
    for label, year_start, year_end in years:
        ends: dict[int, date] = {}
        for end in quarter_ends:
            if end < year_start or (year_end is not None and end >= year_end):
                continue
            quarter = round((end - year_start).days / QUARTER_DAYS)
            if 1 <= quarter <= 3:
                ends[quarter] = end
        if year_end is not None:
            ends[4] = year_end
        for quarter in sorted(ends):
            before = year_start if quarter == 1 else None
            if quarter > 1 and quarter - 1 in ends:
                before = ends[quarter - 1] + timedelta(days=1)
            end = ends[quarter]
            periods.append(Period(label, quarter, before, end, year_start, first_filed.get(end)))
    return periods


def fiscal_year_end_month(periods: list[Period]) -> int:
    """The month fiscal years usually end in (52/53-week years can drift across a month end)."""
    months = Counter(p.end.month for p in periods if p.fiscal_quarter == 4)
    return months.most_common(1)[0][0]


def _quarters_spanned(start: date, end: date) -> int:
    return round(((end - start).days + 1) / QUARTER_DAYS)


def restatement_factor(
    tag: TagFacts, accn: str, quarter: int, last_year: list[Period] | None
) -> Decimal:
    """How a filing scales last year's year-to-date figure against its original: 1 if unchanged.

    0.5 means the filing shows half of what was first filed for the same span, as after a
    spin-off or a 2-for-1 split of a per-share figure. Unknown counts as unchanged.
    """
    prior = next((p for p in last_year or [] if p.fiscal_quarter == quarter), None)
    if prior is None:
        return Decimal(1)
    for start, original in tag.durations.get(prior.end, {}).items():
        same_span = _quarters_spanned(start, prior.end) == quarter
        if not same_span or abs((start - prior.year_start).days) > START_TOLERANCE_DAYS:
            continue
        again = tag.as_filed.get(accn, {}).get((start, prior.end))
        if again is not None and original != 0:
            return again / original
    return Decimal(1)


def bases_differ(now: Decimal, before: Decimal) -> bool:
    """True when two filings' restatement factors show they report on different bases.

    Subtraction is safe when neither filing restated history, or when both restated it alike
    (a split earlier in the year, or the quarters after a spin-off).
    """
    if abs(now - 1) <= RESTATEMENT_TOLERANCE:
        return False
    if abs(before - 1) <= RESTATEMENT_TOLERANCE:
        return True
    return abs(now - before) > SAME_BASIS_TOLERANCE * abs(before)


def quarter_values(
    tag: TagFacts, year: list[Period], last_year: list[Period] | None = None
) -> tuple[dict[int, Decimal], dict[int, Decimal]]:
    """Standalone quarter values of a span-of-time tag for one fiscal year (rules 2 and 5).

    Returns the values, and for each quarter that could not be derived because of a
    restatement, the factor by which its filing restated last year.
    """
    direct: dict[int, Decimal] = {}
    to_date: dict[int, Decimal] = {}
    factor: dict[int, Decimal] = {}
    for period in year:
        for start, value in tag.durations.get(period.end, {}).items():
            quarters = _quarters_spanned(start, period.end)
            starts_with_year = abs((start - period.year_start).days) <= START_TOLERANCE_DAYS
            if quarters == 1:
                direct[period.fiscal_quarter] = value
            if quarters == period.fiscal_quarter and starts_with_year:
                to_date[period.fiscal_quarter] = value
                filing = tag.source.get((start, period.end), "")
                factor[period.fiscal_quarter] = restatement_factor(tag, filing, quarters, last_year)

    values: dict[int, Decimal] = {}
    blocked: dict[int, Decimal] = {}
    for quarter in sorted(p.fiscal_quarter for p in year):
        before = Decimal(0) if quarter == 1 else to_date.get(quarter - 1)
        if quarter in direct:
            values[quarter] = direct[quarter]
            if quarter not in to_date and before is not None:
                to_date[quarter] = before + direct[quarter]
        elif quarter in to_date and before is not None:
            now = factor.get(quarter, Decimal(1))
            if bases_differ(now, factor.get(quarter - 1, Decimal(1))):
                blocked[quarter] = now
            else:
                values[quarter] = to_date[quarter] - before
    return values, blocked


Quarter = tuple[int, int]  # fiscal year, fiscal quarter


def column_values(
    facts: dict[str, Any], tags: list[str], periods: list[Period], point_in_time: bool
) -> tuple[dict[Quarter, Decimal], dict[Quarter, Decimal]]:
    """Values of one column by quarter, and the quarters left empty by a restatement.

    Each quarter uses the first tag with data for it.
    """
    by_year: dict[int, list[Period]] = {}
    for period in periods:
        by_year.setdefault(period.fiscal_year, []).append(period)

    found: dict[Quarter, Decimal] = {}
    blocked: dict[Quarter, Decimal] = {}
    for tag in tags:
        indexed = index_tag(facts, tag)
        for fiscal_year, year in by_year.items():
            if point_in_time:
                values = {
                    p.fiscal_quarter: indexed.instants[p.end]
                    for p in year
                    if p.end in indexed.instants
                }
                unusable: dict[int, Decimal] = {}
            else:
                values, unusable = quarter_values(indexed, year, by_year.get(fiscal_year - 1))
            # A quarter one tag could not derive is not filled from a later, different tag.
            for quarter, factor in unusable.items():
                if (fiscal_year, quarter) not in found:
                    blocked.setdefault((fiscal_year, quarter), factor)
            for quarter, value in values.items():
                if (fiscal_year, quarter) not in blocked:
                    found.setdefault((fiscal_year, quarter), value)
    return found, blocked


Values = dict[str, dict[str, dict[Quarter, Decimal]]]  # table -> column -> quarter


def derive_columns(
    facts: dict[str, Any], inputs: dict[str, list[str]], periods: list[Period], values: Values
) -> set[tuple[str, str, Quarter]]:
    """Calculate what was not filed, in place, and return `(table, column, quarter)` for each.

    - gross_profit = revenue - cost of revenue, only where cost of revenue is filed (so banks,
      which have no cost of revenue, stay NULL).
    - total_liabilities = total liabilities and equity - total equity including noncontrolling
      interest. Anything a company reports between liabilities and equity (redeemable interests)
      therefore counts as a liability.
    - capex = additions to oil and gas properties + additions to other property, for producers
      that file the two lines separately; with only one of them filed it stays NULL.
    A filed value is never replaced.
    """

    def filed(name: str, point_in_time: bool = False) -> dict[Quarter, Decimal]:
        return column_values(facts, inputs.get(name, []), periods, point_in_time)[0]

    derived: set[tuple[str, str, Quarter]] = set()

    def fill(table: str, column: str, calculated: dict[Quarter, Decimal]) -> None:
        target = values.get(table, {}).get(column)
        for key, value in calculated.items():
            if target is not None and key not in target:
                target[key] = value
                derived.add((table, column, key))

    revenue = values.get("income_statements", {}).get("revenue", {})
    cost = filed("cost_of_revenue")
    fill(
        "income_statements",
        "gross_profit",
        {key: revenue[key] - cost[key] for key in revenue.keys() & cost.keys()},
    )
    total = filed("liabilities_and_equity", point_in_time=True)
    equity = filed("total_equity", point_in_time=True)
    fill(
        "balance_sheets",
        "total_liabilities",
        {key: total[key] - equity[key] for key in total.keys() & equity.keys()},
    )
    oil_and_gas, other = filed("capex_oil_and_gas"), filed("capex_other_property")
    fill(
        "cash_flows",
        "capex",
        {key: oil_and_gas[key] + other[key] for key in oil_and_gas.keys() & other.keys()},
    )
    return derived


def build_company(ticker: str, facts: dict[str, Any], mapping: Mapping) -> CompanyData:
    """Turn one company-facts file into rows for the three statement tables."""
    tables = {table: columns for table, columns in mapping.items() if table in KEYS}
    all_tags = [tag for columns in tables.values() for tags in columns.values() for tag in tags]
    calendar = build_calendar(facts, all_tags)
    # The last five completed fiscal years, plus the quarters of the year still in progress.
    completed = max(p.fiscal_year for p in calendar if p.fiscal_quarter == 4)
    periods = [p for p in calendar if p.fiscal_year > completed - FISCAL_YEARS]

    values: Values = {}
    warnings: list[str] = []
    for table, columns in tables.items():
        values[table] = {}
        for column, tags in columns.items():
            found, blocked = column_values(facts, tags, periods, table == "balance_sheets")
            values[table][column] = found
            warnings += [
                f"{table}.{column} left empty for fiscal {year} Q{quarter}: its filing restated "
                f"last year's figure by {(factor - 1) * 100:+.0f}% (spin-off, stock split or "
                "reclassification), so the quarter cannot be derived"
                for (year, quarter), factor in sorted(blocked.items())
                if (year, quarter) not in found
            ]
    derived = derive_columns(facts, mapping.get("derived_inputs", {}), periods, values)

    rows: dict[str, list[Row]] = {}
    missing: list[str] = []
    for table, columns in values.items():
        missing += [f"{table}.{column}" for column, found in columns.items() if not found]
        rows[table] = []
        for period in periods:
            key = (period.fiscal_year, period.fiscal_quarter)
            row: Row = {column: found.get(key) for column, found in columns.items()}
            if all(value is None for value in row.values()):
                continue
            if table == "cash_flows":
                cash, capex = row.get("operating_cash_flow"), row.get("capex")
                both = cash is not None and capex is not None
                row["free_cash_flow"] = cash - capex if both else None
            if table == "income_statements":
                row["period_start"] = period.start
                row["filed_date"] = period.filed
            row |= {
                "ticker": ticker,
                "fiscal_year": period.fiscal_year,
                "fiscal_quarter": period.fiscal_quarter,
                "period_end": period.end,
                "derived_columns": sorted(c for c in columns if (table, c, key) in derived),
            }
            rows[table].append(row)
    return CompanyData(fiscal_year_end_month(calendar), rows, missing, warnings)


def _facts_by_filing(facts: dict[str, Any]) -> dict[str, list[tuple[str, str, dict[str, Any]]]]:
    """Every us-gaap fact as `(tag, unit, fact)`, grouped by the filing it came from."""
    filings: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for tag, body in facts.get("facts", {}).get("us-gaap", {}).items():
        for unit, rows in body.get("units", {}).items():
            for fact in rows:
                filings.setdefault(fact.get("accn", ""), []).append((tag, unit, fact))
    return filings


def merge_related(
    main: dict[str, Any], other: dict[str, Any], anchor_tags: list[str], require_match: bool
) -> int:
    """Add to `main` the filings only `other` has, and return how many were added (rule 6).

    A predecessor's filings are all taken. A co-registrant's file also holds that subsidiary's
    own numbers, so with `require_match` a filing is taken only when it repeats a total-assets
    figure the company already reported for the same date (every 10-Q restates the last
    year-end balance sheet). Filings accepted this way can vouch for further ones.
    """

    def anchors(rows: list[tuple[str, str, dict[str, Any]]]) -> set[tuple[str, Any]]:
        return {
            (fact["end"], fact["val"])
            for tag, _, fact in rows
            if tag in anchor_tags and "start" not in fact
        }

    existing = _facts_by_filing(main)
    candidates = {a: rows for a, rows in _facts_by_filing(other).items() if a not in existing}
    adopted = candidates
    if require_match:
        known = set().union(*(anchors(rows) for rows in existing.values()))
        adopted = {}
        found_one = True
        while found_one:
            found_one = False
            for accn in [a for a, rows in candidates.items() if anchors(rows) & known]:
                adopted[accn] = candidates.pop(accn)
                known |= anchors(adopted[accn])
                found_one = True

    gaap = main.setdefault("facts", {}).setdefault("us-gaap", {})
    for rows in adopted.values():
        for tag, unit, fact in rows:
            gaap.setdefault(tag, {}).setdefault("units", {}).setdefault(unit, []).append(fact)
    return len(adopted)


# --- Database ---------------------------------------------------------------------------------


def upsert(conn: Connection, table_name: str, rows: list[Row]) -> int:
    """Insert or update rows and return how many were actually new or different."""
    if not rows:
        return 0
    table = Table(table_name, MetaData(), autoload_with=conn)
    keys = KEYS[table_name]
    changing = [name for name in rows[0] if name not in keys]
    statement = pg_insert(table).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=keys,
        set_={name: statement.excluded[name] for name in changing},
        # Leaves identical rows untouched, so a second run reports nothing changed.
        where=tuple_(*(table.c[name] for name in changing)).is_distinct_from(
            tuple_(*(statement.excluded[name] for name in changing))
        ),
    ).returning(table.c.ticker)
    return len(conn.execute(statement).all())


def store_company(engine: Engine, ticker: str, name: str, cik: int, data: CompanyData) -> int:
    """Write one company and its statements in a single transaction; return rows changed."""
    company = {
        "ticker": ticker,
        "name": name,
        "cik": cik,
        "fiscal_year_end_month": data.fiscal_year_end_month,
    }
    with engine.begin() as conn:
        changed = upsert(conn, "companies", [company])
        for table, rows in data.rows.items():
            changed += upsert(conn, table, rows)
    return changed


# --- Command line -----------------------------------------------------------------------------


def load(
    engine: Engine,
    client: SecClient,
    companies: list[tuple[str, str]],
    mapping: Mapping,
    refresh: bool = False,
    raw_dir: Path = RAW_DIR,
    related: dict[str, dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
    """Load every company and return the report. One company's failure does not stop the rest."""
    tickers = fetch_json(client, TICKERS_URL, raw_dir / "company_tickers.json", refresh)
    index = cik_by_ticker(tickers)
    report: dict[str, Any] = {}
    for ticker, name in companies:
        try:
            cik = resolve_cik(ticker, index)
            facts = fetch_json(
                client, FACTS_URL.format(cik=cik), raw_dir / f"{ticker}.json", refresh
            )
            extra_filings = 0
            for kind in ("predecessors", "co_registrants"):
                for other_cik in (related or {}).get(ticker, {}).get(kind, []):
                    other = fetch_json(
                        client,
                        FACTS_URL.format(cik=other_cik),
                        raw_dir / f"{ticker}_CIK{other_cik:010d}.json",
                        refresh,
                    )
                    extra_filings += merge_related(
                        facts,
                        other,
                        mapping["balance_sheets"]["total_assets"],
                        require_match=kind == "co_registrants",
                    )
            data = build_company(ticker, facts, mapping)
            changed = store_company(engine, ticker, name, cik, data)
        except SecError as exc:
            logger.error("%s: %s", ticker, exc)
            report[ticker] = {"error": str(exc)}
            continue
        report[ticker] = {
            "cik": cik,
            "rows": {table: len(rows) for table, rows in data.rows.items()},
            "changed_rows": changed,
            "filings_from_related_ciks": extra_filings,
            "derived_values": {
                table: sum(bool(row["derived_columns"]) for row in rows)
                for table, rows in data.rows.items()
            },
            "missing_columns": data.missing_columns,
            "warnings": data.warnings,
        }
        if data.missing_columns:
            logger.warning("%s: no tag found for %s", ticker, ", ".join(data.missing_columns))
        for warning in data.warnings:
            logger.warning("%s: %s", ticker, warning)
        logger.info("%s: %d rows changed", ticker, changed)
    return report


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    """Save the per-company summary, including every column that found no tag."""
    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "companies_loaded": sum("error" not in entry for entry in report.values()),
        "companies_failed": sum("error" in entry for entry in report.values()),
        "changed_rows": sum(entry.get("changed_rows", 0) for entry in report.values()),
        "companies": report,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    from app.config import get_settings
    from app.db.session import admin_engine

    parser = argparse.ArgumentParser(description="Load quarterly financials from SEC EDGAR.")
    parser.add_argument("--tickers", help="comma-separated subset of ingestion/companies.csv")
    parser.add_argument("--refresh", action="store_true", help="download again, ignoring the cache")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        companies = read_companies()
        if args.tickers:
            wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
            unknown = wanted - {ticker for ticker, _ in companies}
            if unknown:
                raise SecError(f"not in companies.csv: {', '.join(sorted(unknown))}")
            companies = [c for c in companies if c[0] in wanted]
        if not companies:
            raise SecError("ingestion/companies.csv has no companies (columns: ticker,name)")
        client = SecClient(get_settings().sec_user_agent)
        report = load(
            admin_engine,
            client,
            companies,
            read_mapping(),
            refresh=args.refresh,
            related=read_related(),
        )
    except SecError as exc:
        logger.error("%s", exc)
        return 1
    except OperationalError:
        # The driver's message can include the connection target; keep it out of the log.
        logger.error("database unreachable; is `docker compose up -d db` running?")
        return 1

    write_report(report)
    failed = sum("error" in entry for entry in report.values())
    changed = sum(entry.get("changed_rows", 0) for entry in report.values())
    logger.info(
        "%d companies loaded, %d failed, %d changed rows", len(report) - failed, failed, changed
    )
    return 1 if failed == len(report) else 0


if __name__ == "__main__":
    sys.exit(main())
