"""Loads daily prices, dividends, stock splits and sectors from yfinance.

Run with `uv run python -m ingestion.load_prices [--tickers AAPL,MSFT] [--refresh]`.

What is stored
--------------
- `daily_prices`: five years of daily bars for every ticker in `companies`. `close` is as Yahoo
  gives it, already adjusted for stock splits; `adj_close` is also adjusted for dividends. Yahoo
  recalculates `adj_close` for the whole history after each dividend, so a download on a later
  day can change old rows. Today's bar (New York date) is left out, because the session may not
  be finished.
- `dividends`: the payouts with an ex-date inside the same five years (they come with the prices,
  in one request).
- `stock_splits`: every split Yahoo lists for the company, as new shares per old share (10 for
  a 10-for-1 split, 0.125 for a 1-for-8 reverse split). The full history is kept, because the
  view `income_statements_split_adjusted` needs the splits after each quarter was filed, and the
  first fiscal quarters in the data are older than the price window. Yahoo also lists some
  spin-offs as splits with an odd ratio (GE: 1.281); they are stored as given, since `close`
  is adjusted for them too.
- `companies.sector` and `companies.industry`: Yahoo's own names ("Financial Services"), which
  are not GICS names. A value Yahoo does not give is left as it is in the database.

Downloads are cached under `data/raw/` (`prices/`, `dividends/`, `splits/`, `profiles/`) and
reused for one day. Rows are always read back from the cache files, so a second run stores
exactly what the first did and changes nothing.
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import OperationalError
from yfinance.exceptions import YFRateLimitError

from app.config import LOG_FORMAT
from ingestion.load_sec import ROOT, Row, is_fresh, upsert

RAW_DIR = ROOT / "data" / "raw"

HISTORY_PERIOD = "5y"
CACHE_MAX_AGE = timedelta(days=1)
REQUEST_PAUSE = 0.5  # seconds between requests to Yahoo
MAX_ATTEMPTS = 3
RETRY_WAIT = 1.0  # seconds before retrying a failed request; doubles each time
RATE_LIMIT_WAIT = 10.0  # the same, after Yahoo answers "too many requests"
UPSERT_CHUNK = 2000  # rows per statement, well under the driver's 65,535 parameter limit
CENTS = Decimal("0.0001")
RATIO_PLACES = Decimal("0.000001")
MARKET_TIMEZONE = ZoneInfo("America/New_York")

PRICE_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj_close": "Adj Close",
}
PRICE_FIELDS = ["trade_date", *PRICE_COLUMNS, "volume"]
DIVIDEND_FIELDS = ["ex_date", "amount"]
SPLIT_FIELDS = ["split_date", "ratio"]
DATE_FIELDS = {"trade_date", "ex_date", "split_date"}
PROFILE_FIELDS = ("sector", "industry")

logger = logging.getLogger(__name__)

Profile = dict[str, str | None]


class PriceError(Exception):
    """Price data could not be fetched or understood."""


class Source(Protocol):
    """Where prices and profiles come from; tests pass a fake."""

    def history(self, ticker: str) -> pd.DataFrame: ...

    def splits(self, ticker: str) -> pd.Series: ...

    def profile(self, ticker: str) -> Profile: ...


# --- Yahoo ------------------------------------------------------------------------------------


class YahooSource:
    """The only code that calls yfinance. Paces requests and retries failed ones."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self._sleep = sleep
        self._requests = 0
        self._blocked = False
        # By default yfinance logs most failures and returns nothing; raising them lets a
        # passing failure be retried and a lasting one be reported with its cause.
        yf.config.debug.hide_exceptions = False

    def _call[T](self, ticker: str, what: str, fetch: Callable[[yf.Ticker], T]) -> T:
        if self._blocked:
            # Yahoo is still refusing; do not wait through the retries again for every ticker.
            raise PriceError(f"{what} not requested: Yahoo rate limited an earlier request")
        symbol = ticker.replace(".", "-")  # Yahoo writes share classes as BRK-B
        wait = REQUEST_PAUSE if self._requests else 0.0
        problem: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            if wait:
                self._sleep(wait)
            self._requests += 1
            try:
                return fetch(yf.Ticker(symbol))
            except YFRateLimitError as exc:
                problem, wait = exc, RATE_LIMIT_WAIT * 2**attempt
            except Exception as exc:
                # yfinance is unofficial and raises many unrelated error types (HTTP, parsing,
                # KeyError on a changed page); each is retried, then reported for this ticker.
                problem, wait = exc, RETRY_WAIT * 2**attempt
        if isinstance(problem, YFRateLimitError):
            self._blocked = True
            raise PriceError(f"{what} request was rate limited {MAX_ATTEMPTS} times") from problem
        cause = type(problem).__name__
        raise PriceError(f"{what} request failed {MAX_ATTEMPTS} times ({cause})") from problem

    def history(self, ticker: str) -> pd.DataFrame:
        """Five years of daily bars with `Adj Close` and `Dividends` columns."""
        return self._call(
            ticker,
            "price history",
            lambda t: t.history(
                period=HISTORY_PERIOD, interval="1d", auto_adjust=False, actions=True
            ),
        )

    def splits(self, ticker: str) -> pd.Series:
        """Every split Yahoo lists, as ratios indexed by date. This is a second, longer request."""
        return self._call(ticker, "split history", lambda t: t.get_splits(period="max"))

    def profile(self, ticker: str) -> Profile:
        """Yahoo's sector and industry, each None when Yahoo has none."""
        info = self._call(ticker, "profile", lambda t: t.info) or {}
        return {name: (info.get(name) or None) for name in PROFILE_FIELDS}


# --- Cleaning ---------------------------------------------------------------------------------


def to_decimal(value: Any) -> Decimal | None:
    """A float from pandas as a 4-place Decimal, or None for a missing value."""
    if pd.isna(value):
        return None
    return Decimal(str(float(value))).quantize(CENTS)


def price_rows(frame: pd.DataFrame, today: date) -> list[dict[str, Any]]:
    """Daily bars from a yfinance frame, without today's and without days that have no close."""
    missing = [name for name in [*PRICE_COLUMNS.values(), "Volume"] if name not in frame.columns]
    if missing:
        raise PriceError(f"price history has no {', '.join(missing)} column")
    rows: list[dict[str, Any]] = []
    for stamp, bar in frame.iterrows():
        trade_date = stamp.date()
        if trade_date >= today or pd.isna(bar["Close"]):
            continue
        row: dict[str, Any] = {"trade_date": trade_date}
        row |= {column: to_decimal(bar[source]) for column, source in PRICE_COLUMNS.items()}
        row["volume"] = None if pd.isna(bar["Volume"]) else int(bar["Volume"])
        rows.append(row)
    return rows


def dividend_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The payouts in a yfinance frame: the days whose `Dividends` value is above zero."""
    if "Dividends" not in frame.columns:
        return []
    paid = frame["Dividends"]
    return [
        {"ex_date": stamp.date(), "amount": to_decimal(amount)}
        for stamp, amount in paid[paid > 0].items()
    ]


def split_rows(splits: pd.Series) -> list[dict[str, Any]]:
    """Splits from a yfinance series: new shares per old share, on the day it took effect."""
    return [
        {"split_date": stamp.date(), "ratio": Decimal(str(float(ratio))).quantize(RATIO_PLACES)}
        for stamp, ratio in splits.items()
        if not pd.isna(ratio) and ratio > 0
    ]


def market_today() -> date:
    """Today's date where the exchanges are."""
    return datetime.now(tz=MARKET_TIMEZONE).date()


# --- Cache ------------------------------------------------------------------------------------


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse(field: str, value: str) -> Any:
    if value == "":
        return None
    if field in DATE_FIELDS:
        return date.fromisoformat(value)
    return int(value) if field == "volume" else Decimal(value)


def read_csv(path: Path, fields: list[str]) -> list[dict[str, Any]]:
    """Rows of a cache file, typed for the database."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise PriceError(f"{path.name} does not have the columns {','.join(fields)}")
        try:
            return [{name: _parse(name, line[name]) for name in fields} for line in reader]
        except (ValueError, ArithmeticError) as exc:
            raise PriceError(f"{path.name} has a value that cannot be read") from exc


def _same_bar(new: dict[str, Any], old: dict[str, Any]) -> bool:
    if new["volume"] != old["volume"]:
        return False
    for name in PRICE_COLUMNS:
        if new[name] is None or old[name] is None:
            if new[name] is not old[name]:
                return False
        elif abs(new[name] - old[name]) > CENTS:
            return False
    return True


def keep_settled(new: list[dict[str, Any]], old: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace each new bar with the cached one for that day when they differ by rounding only.

    Yahoo's floats change in the fifth decimal from one request to the next, which flips the
    last stored digit of about one `adj_close` in twelve. Without this, every download would
    rewrite thousands of rows that have not really changed. A real change (the adjustment
    after a dividend, a corrected volume) is far larger than 0.0001 and is kept.
    """
    cached = {row["trade_date"]: row for row in old}
    return [
        cached[row["trade_date"]]
        if row["trade_date"] in cached and _same_bar(row, cached[row["trade_date"]])
        else row
        for row in new
    ]


def cached_history(
    source: Source, ticker: str, raw_dir: Path, refresh: bool, today: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Price and dividend rows for a ticker, downloaded only when the cache is stale."""
    prices_path = raw_dir / "prices" / f"{ticker}.csv"
    dividends_path = raw_dir / "dividends" / f"{ticker}.csv"
    now = datetime.now(tz=UTC)
    fresh = all(is_fresh(path, now, CACHE_MAX_AGE) for path in (prices_path, dividends_path))
    if refresh or not fresh:
        frame = source.history(ticker)
        prices = price_rows(frame, today) if not frame.empty else []
        if not prices:
            raise PriceError("Yahoo returned no prices")
        if prices_path.exists():
            try:
                prices = keep_settled(prices, read_csv(prices_path, PRICE_FIELDS))
            except PriceError:
                logger.warning("%s: old price cache is unreadable and is replaced", ticker)
        write_csv(prices_path, PRICE_FIELDS, prices)
        write_csv(dividends_path, DIVIDEND_FIELDS, dividend_rows(frame))
    return read_csv(prices_path, PRICE_FIELDS), read_csv(dividends_path, DIVIDEND_FIELDS)


def cached_splits(
    source: Source, ticker: str, raw_dir: Path, refresh: bool
) -> list[dict[str, Any]]:
    """Split rows for a ticker, downloaded only when the cache is stale."""
    path = raw_dir / "splits" / f"{ticker}.csv"
    if refresh or not is_fresh(path, datetime.now(tz=UTC), CACHE_MAX_AGE):
        write_csv(path, SPLIT_FIELDS, split_rows(source.splits(ticker)))
    return read_csv(path, SPLIT_FIELDS)


def cached_profile(source: Source, ticker: str, raw_dir: Path, refresh: bool) -> Profile:
    """Sector and industry for a ticker, downloaded only when the cache is stale."""
    path = raw_dir / "profiles" / f"{ticker}.json"
    if not refresh and is_fresh(path, datetime.now(tz=UTC), CACHE_MAX_AGE):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PriceError(f"{path.name} is not valid JSON") from exc
        return {name: stored.get(name) for name in PROFILE_FIELDS}
    profile = source.profile(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile) + "\n", encoding="utf-8")
    return profile


# --- Database ---------------------------------------------------------------------------------

UPDATE_PROFILE = text(
    """
    UPDATE companies
    SET sector = COALESCE(CAST(:sector AS TEXT), sector),
        industry = COALESCE(CAST(:industry AS TEXT), industry)
    WHERE ticker = :ticker
      AND (COALESCE(CAST(:sector AS TEXT), sector) IS DISTINCT FROM sector
           OR COALESCE(CAST(:industry AS TEXT), industry) IS DISTINCT FROM industry)
    """
)


def read_tickers(engine: Engine) -> list[str]:
    """Every ticker in `companies`, which the SEC loader fills."""
    with engine.connect() as conn:
        return list(conn.execute(text("SELECT ticker FROM companies ORDER BY ticker")).scalars())


def update_profile(conn: Connection, ticker: str, profile: Profile) -> int:
    """Set sector and industry where Yahoo gave one; a missing value never erases a stored one."""
    values = {"ticker": ticker} | {name: profile.get(name) for name in PROFILE_FIELDS}
    return conn.execute(UPDATE_PROFILE, values).rowcount


def store_ticker(
    engine: Engine,
    ticker: str,
    prices: list[Row],
    dividends: list[Row],
    splits: list[Row],
    profile: Profile,
) -> int:
    """Write one ticker's rows in a single transaction; return how many rows changed."""
    changed = 0
    with engine.begin() as conn:
        for start in range(0, len(prices), UPSERT_CHUNK):
            chunk = prices[start : start + UPSERT_CHUNK]
            changed += upsert(conn, "daily_prices", [{"ticker": ticker} | row for row in chunk])
        changed += upsert(conn, "dividends", [{"ticker": ticker} | row for row in dividends])
        changed += upsert(conn, "stock_splits", [{"ticker": ticker} | row for row in splits])
        changed += update_profile(conn, ticker, profile)
    return changed


def table_counts(engine: Engine) -> dict[str, int]:
    """Rows now in each table this loader fills."""
    with engine.connect() as conn:
        return {
            "daily_prices": conn.execute(text("SELECT count(*) FROM daily_prices")).scalar_one(),
            "dividends": conn.execute(text("SELECT count(*) FROM dividends")).scalar_one(),
            "stock_splits": conn.execute(text("SELECT count(*) FROM stock_splits")).scalar_one(),
            "companies_with_sector": conn.execute(
                text("SELECT count(*) FROM companies WHERE sector IS NOT NULL")
            ).scalar_one(),
        }


# --- Command line -----------------------------------------------------------------------------


def load(
    engine: Engine,
    source: Source,
    tickers: list[str],
    refresh: bool = False,
    raw_dir: Path = RAW_DIR,
    today: date | None = None,
) -> dict[str, Any]:
    """Load every ticker and return the report. One ticker's failure does not stop the rest."""
    today = today or market_today()
    report: dict[str, Any] = {}
    for ticker in tickers:
        try:
            prices, dividends = cached_history(source, ticker, raw_dir, refresh, today)
            splits = cached_splits(source, ticker, raw_dir, refresh)
            try:
                profile = cached_profile(source, ticker, raw_dir, refresh)
                profile_error = None
            except PriceError as exc:
                # Prices are still worth loading without a sector.
                profile, profile_error = dict.fromkeys(PROFILE_FIELDS), str(exc)
            changed = store_ticker(engine, ticker, prices, dividends, splits, profile)
        except PriceError as exc:
            logger.error("%s: %s", ticker, exc)
            report[ticker] = {"error": str(exc)}
            continue
        missing = [name for name in PROFILE_FIELDS if profile.get(name) is None]
        report[ticker] = {
            "prices": len(prices),
            "dividends": len(dividends),
            "splits": len(splits),
            "first_date": prices[0]["trade_date"].isoformat(),
            "last_date": prices[-1]["trade_date"].isoformat(),
            "changed_rows": changed,
            "missing_profile": missing,
        }
        if profile_error:
            logger.warning("%s: %s; sector and industry left as they were", ticker, profile_error)
        elif missing:
            logger.warning("%s: Yahoo gives no %s", ticker, " or ".join(missing))
        logger.info(
            "%s: %d prices, %d dividends, %d splits, %d rows changed",
            ticker,
            len(prices),
            len(dividends),
            len(splits),
            changed,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    from app.db.session import admin_engine

    parser = argparse.ArgumentParser(
        description="Load prices, dividends, splits and sectors (yfinance)."
    )
    parser.add_argument("--tickers", help="comma-separated subset of the companies table")
    parser.add_argument("--refresh", action="store_true", help="download again, ignoring the cache")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    # yfinance logs each failure itself, not as JSON; failures are reported below instead.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    try:
        tickers = read_tickers(admin_engine)
        if not tickers:
            raise PriceError("the companies table is empty; run `python -m ingestion.load_sec`")
        if args.tickers:
            wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
            unknown = wanted - set(tickers)
            if unknown:
                raise PriceError(f"not in the companies table: {', '.join(sorted(unknown))}")
            tickers = [t for t in tickers if t in wanted]
        report = load(admin_engine, YahooSource(), tickers, refresh=args.refresh)
        totals = table_counts(admin_engine)
    except PriceError as exc:
        logger.error("%s", exc)
        return 1
    except OperationalError:
        # The driver's message can include the connection target; keep it out of the log.
        logger.error("database unreachable; is `docker compose up -d db` running?")
        return 1

    failed = sorted(ticker for ticker, entry in report.items() if "error" in entry)
    no_sector = sorted(t for t, e in report.items() if "sector" in e.get("missing_profile", []))
    changed = sum(entry.get("changed_rows", 0) for entry in report.values())
    logger.info(
        "%d tickers loaded, %d failed, %d changed rows",
        len(report) - len(failed),
        len(failed),
        changed,
    )
    logger.info(
        "rows now: daily_prices %d, dividends %d, stock_splits %d; companies with a sector: %d",
        totals["daily_prices"],
        totals["dividends"],
        totals["stock_splits"],
        totals["companies_with_sector"],
    )
    if failed:
        logger.error("failed tickers: %s", ", ".join(failed))
    if no_sector:
        logger.warning("tickers without a sector from Yahoo: %s", ", ".join(no_sector))
    return 1 if len(failed) == len(report) else 0


if __name__ == "__main__":
    sys.exit(main())
