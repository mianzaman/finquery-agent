"""Tests for the prices loader. yfinance is replaced everywhere; nothing touches the network."""

import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import text
from yfinance.exceptions import YFRateLimitError

from ingestion import load_all, load_prices, load_sec
from ingestion.load_prices import (
    MAX_ATTEMPTS,
    PRICE_FIELDS,
    PriceError,
    Profile,
    YahooSource,
    cached_history,
    cached_profile,
    cached_splits,
    dividend_rows,
    load,
    price_rows,
    read_csv,
    split_rows,
    store_ticker,
    to_decimal,
    update_profile,
)

TODAY = date(2026, 11, 13)
NAN = float("nan")


def frame(*days: tuple[str, float, float]) -> pd.DataFrame:
    """A yfinance-style frame from `(date, close, dividend)`; other prices follow the close."""
    index = pd.DatetimeIndex([day for day, _, _ in days], tz="America/New_York", name="Date")
    closes = [close for _, close, _ in days]
    return pd.DataFrame(
        {
            "Open": [c - 1 for c in closes],
            "High": [c + 2 for c in closes],
            "Low": [c - 2 for c in closes],
            "Close": closes,
            "Adj Close": [c * 0.9 for c in closes],
            "Volume": [1000 + i for i in range(len(days))],
            "Dividends": [dividend for _, _, dividend in days],
            "Stock Splits": [0.0] * len(days),
        },
        index=index,
    )


def split_series(*splits: tuple[str, float]) -> pd.Series:
    """A yfinance-style split series from `(date, ratio)`."""
    index = pd.DatetimeIndex([day for day, _ in splits], tz="America/New_York", name="Date")
    return pd.Series([ratio for _, ratio in splits], index=index, name="Stock Splits", dtype=float)


SAMPLE = frame(
    ("2026-11-09", 100.0, 0.0),
    ("2026-11-10", 101.123456, 0.25),
    ("2026-11-11", 102.5, 0.0),
    ("2026-11-12", 103.0, 0.0),
    ("2026-11-13", 104.0, 0.0),  # today: an unfinished session
)


class FakeSource:
    """Stands in for Yahoo and records what was asked."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        profiles: dict[str, Profile] | None = None,
        no_profile: tuple[str, ...] = (),
        split_history: dict[str, pd.Series] | None = None,
    ) -> None:
        self.frames = frames
        self.split_history = split_history or {}
        self.split_calls: list[str] = []
        self.profiles = profiles or {}
        self.no_profile = no_profile
        self.history_calls: list[str] = []
        self.profile_calls: list[str] = []

    def history(self, ticker: str) -> pd.DataFrame:
        self.history_calls.append(ticker)
        if ticker not in self.frames:
            raise PriceError("price history request failed (KeyError)")
        return self.frames[ticker]

    def splits(self, ticker: str) -> pd.Series:
        self.split_calls.append(ticker)
        return self.split_history.get(ticker, split_series())

    def profile(self, ticker: str) -> Profile:
        self.profile_calls.append(ticker)
        if ticker in self.no_profile:
            raise PriceError("profile request failed (HTTPError)")
        return self.profiles.get(ticker, {"sector": None, "industry": None})


# --- Cleaning ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (101.123456, Decimal("101.1235")),
        (0.1 + 0.2, Decimal("0.3000")),
        (150, Decimal("150.0000")),
        (NAN, None),
        (None, None),
    ],
)
def test_to_decimal(value: Any, expected: Decimal | None) -> None:
    assert to_decimal(value) == expected


def test_price_rows_are_typed_and_rounded() -> None:
    rows = price_rows(SAMPLE, TODAY)

    assert list(rows[1]) == PRICE_FIELDS
    assert rows[1] == {
        "trade_date": date(2026, 11, 10),
        "open": Decimal("100.1235"),
        "high": Decimal("103.1235"),
        "low": Decimal("99.1235"),
        "close": Decimal("101.1235"),
        "adj_close": Decimal("91.0111"),
        "volume": 1001,
    }


def test_price_rows_leave_out_today_and_days_without_a_close() -> None:
    with_gap = frame(("2026-11-10", 50.0, 0.0), ("2026-11-11", NAN, 0.0), ("2026-11-12", 51.0, 0.0))
    with_gap.loc[with_gap.index[2], "Volume"] = NAN

    assert [r["trade_date"].day for r in price_rows(SAMPLE, TODAY)] == [9, 10, 11, 12]
    rows = price_rows(with_gap, TODAY)
    assert [r["trade_date"].day for r in rows] == [10, 12]
    assert rows[1]["volume"] is None


def test_price_rows_reject_a_frame_without_adjusted_close() -> None:
    with pytest.raises(PriceError, match="Adj Close"):
        price_rows(SAMPLE.drop(columns=["Adj Close"]), TODAY)


def test_dividend_rows_are_the_days_with_a_payout() -> None:
    assert dividend_rows(SAMPLE) == [{"ex_date": date(2026, 11, 10), "amount": Decimal("0.2500")}]
    assert dividend_rows(frame(("2026-11-10", 50.0, 0.0))) == []
    assert dividend_rows(SAMPLE.drop(columns=["Dividends"])) == []


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(10.0, "10.000000"), (0.125, "0.125000"), (1.5, "1.500000"), (1.281, "1.281000")],
)
def test_split_rows_keep_the_ratio_as_new_shares_per_old(ratio: float, expected: str) -> None:
    rows = split_rows(split_series(("2024-06-10", ratio)))

    assert rows == [{"split_date": date(2024, 6, 10), "ratio": Decimal(expected)}]
    assert str(rows[0]["ratio"]) == expected


def test_split_rows_skip_empty_values() -> None:
    assert split_rows(split_series()) == []
    assert split_rows(split_series(("2024-06-10", 0.0), ("2024-06-11", NAN))) == []


# --- Yahoo ------------------------------------------------------------------------------------


class FakeTicker:
    """Replaces `yfinance.Ticker`. `failures` are raised first, one per request."""

    symbols: list[str] = []
    history_arguments: list[dict[str, Any]] = []
    split_arguments: list[dict[str, Any]] = []
    failures: list[Exception] = []
    frame: pd.DataFrame = SAMPLE
    information: dict[str, Any] | None = None

    def __init__(self, symbol: str) -> None:
        FakeTicker.symbols.append(symbol)

    def history(self, **arguments: Any) -> pd.DataFrame:
        FakeTicker.history_arguments.append(arguments)
        if FakeTicker.failures:
            raise FakeTicker.failures.pop(0)
        return FakeTicker.frame

    def get_splits(self, **arguments: Any) -> pd.Series:
        FakeTicker.split_arguments.append(arguments)
        if FakeTicker.failures:
            raise FakeTicker.failures.pop(0)
        return split_series(("2024-06-10", 10.0))

    @property
    def info(self) -> dict[str, Any] | None:
        if FakeTicker.failures:
            raise FakeTicker.failures.pop(0)
        return FakeTicker.information


@pytest.fixture
def yahoo(monkeypatch: pytest.MonkeyPatch) -> type[FakeTicker]:
    FakeTicker.symbols = []
    FakeTicker.history_arguments = []
    FakeTicker.split_arguments = []
    FakeTicker.failures = []
    FakeTicker.information = {"sector": "Technology", "industry": "Consumer Electronics"}
    monkeypatch.setattr(load_prices.yf, "Ticker", FakeTicker)
    return FakeTicker


def test_source_asks_for_five_years_of_unadjusted_daily_bars(yahoo: type[FakeTicker]) -> None:
    result = YahooSource(sleep=lambda _: None).history("BRK.B")

    assert result is SAMPLE
    assert yahoo.symbols == ["BRK-B"]
    assert yahoo.history_arguments == [
        {"period": "5y", "interval": "1d", "auto_adjust": False, "actions": True}
    ]


def test_source_asks_for_the_whole_split_history(yahoo: type[FakeTicker]) -> None:
    result = YahooSource(sleep=lambda _: None).splits("BRK.B")

    assert split_rows(result) == [{"split_date": date(2024, 6, 10), "ratio": Decimal(10)}]
    assert yahoo.symbols == ["BRK-B"]
    assert yahoo.split_arguments == [{"period": "max"}]


@pytest.mark.parametrize(
    ("information", "expected"),
    [
        (
            {"sector": "Technology", "industry": "Consumer Electronics", "other": 1},
            {"sector": "Technology", "industry": "Consumer Electronics"},
        ),
        ({"sector": "", "longName": "X"}, {"sector": None, "industry": None}),
        (None, {"sector": None, "industry": None}),
    ],
)
def test_source_profile(
    yahoo: type[FakeTicker], information: dict[str, Any] | None, expected: Profile
) -> None:
    yahoo.information = information
    assert YahooSource(sleep=lambda _: None).profile("AAPL") == expected


def test_source_pauses_between_requests_and_retries_a_rate_limit(yahoo: type[FakeTicker]) -> None:
    sleeps: list[float] = []
    source = YahooSource(sleep=sleeps.append)
    yahoo.failures = [YFRateLimitError()]

    source.history("AAPL")
    source.profile("AAPL")

    assert sleeps == [load_prices.RATE_LIMIT_WAIT, load_prices.REQUEST_PAUSE]
    assert len(yahoo.symbols) == 3


def test_source_gives_up_on_rate_limits_and_stops_asking(yahoo: type[FakeTicker]) -> None:
    sleeps: list[float] = []
    source = YahooSource(sleep=sleeps.append)
    yahoo.failures = [YFRateLimitError() for _ in range(MAX_ATTEMPTS)]

    with pytest.raises(PriceError, match="rate limited"):
        source.history("AAPL")
    with pytest.raises(PriceError, match="rate limited"):
        source.history("MSFT")

    assert len(yahoo.symbols) == MAX_ATTEMPTS  # the second ticker was never requested
    assert len(sleeps) == MAX_ATTEMPTS - 1


def test_source_retries_a_passing_failure(yahoo: type[FakeTicker]) -> None:
    sleeps: list[float] = []
    yahoo.failures = [ConnectionError("reset"), KeyError("chart")]

    assert YahooSource(sleep=sleeps.append).history("AAPL") is SAMPLE
    assert sleeps == [load_prices.RETRY_WAIT, load_prices.RETRY_WAIT * 2]


def test_source_turns_a_lasting_yfinance_error_into_a_price_error(yahoo: type[FakeTicker]) -> None:
    yahoo.failures = [KeyError("quoteSummary") for _ in range(MAX_ATTEMPTS)]
    source = YahooSource(sleep=lambda _: None)

    with pytest.raises(PriceError, match="profile request failed 3 times \\(KeyError\\)"):
        source.profile("AAPL")
    # Unlike a rate limit, this does not stop later requests.
    assert source.profile("MSFT")["sector"] == "Technology"


# --- Cache ------------------------------------------------------------------------------------


def age(path: Path, days: float) -> None:
    old = (datetime.now(tz=UTC) - timedelta(days=days)).timestamp()
    os.utime(path, (old, old))


def test_history_cache_is_used_until_it_is_a_day_old(tmp_path: Path) -> None:
    source = FakeSource({"ZZPRCA": SAMPLE})

    first = cached_history(source, "ZZPRCA", tmp_path, False, TODAY)
    second = cached_history(source, "ZZPRCA", tmp_path, False, TODAY)
    assert source.history_calls == ["ZZPRCA"]
    assert first == second
    assert len(first[0]) == 4 and len(first[1]) == 1
    assert (tmp_path / "prices" / "ZZPRCA.csv").exists()
    assert (tmp_path / "dividends" / "ZZPRCA.csv").exists()

    cached_history(source, "ZZPRCA", tmp_path, True, TODAY)
    assert len(source.history_calls) == 2  # --refresh

    age(tmp_path / "prices" / "ZZPRCA.csv", 2)
    cached_history(source, "ZZPRCA", tmp_path, False, TODAY)
    assert len(source.history_calls) == 3  # stale


def test_cached_rows_match_what_was_downloaded(tmp_path: Path) -> None:
    prices, dividends = cached_history(
        FakeSource({"ZZPRCA": SAMPLE}), "ZZPRCA", tmp_path, False, TODAY
    )

    assert prices == price_rows(SAMPLE, TODAY)
    assert dividends == dividend_rows(SAMPLE)


def test_a_company_without_dividends_still_gets_a_cache_file(tmp_path: Path) -> None:
    source = FakeSource({"ZZPRCA": frame(("2026-11-10", 50.0, 0.0))})

    assert cached_history(source, "ZZPRCA", tmp_path, False, TODAY)[1] == []
    assert cached_history(source, "ZZPRCA", tmp_path, False, TODAY)[1] == []
    assert source.history_calls == ["ZZPRCA"]


@pytest.mark.parametrize("empty", [SAMPLE.iloc[0:0], frame(("2026-11-13", 104.0, 0.0))])
def test_no_prices_is_an_error_and_writes_no_cache(tmp_path: Path, empty: pd.DataFrame) -> None:
    with pytest.raises(PriceError, match="no prices"):
        cached_history(FakeSource({"ZZPRCA": empty}), "ZZPRCA", tmp_path, False, TODAY)
    assert not (tmp_path / "prices" / "ZZPRCA.csv").exists()


def test_a_new_download_keeps_cached_values_that_differ_by_rounding_only(tmp_path: Path) -> None:
    cached_history(FakeSource({"ZZPRCA": SAMPLE}), "ZZPRCA", tmp_path, False, TODAY)
    again = SAMPLE.copy()
    again.loc[again.index[0], "Adj Close"] = 90.00008  # float noise: 90.0000 -> 90.0001
    again.loc[again.index[1], "Adj Close"] = 90.5  # a real change after a dividend
    again.loc[again.index[2], "Volume"] = 5

    prices, _ = cached_history(FakeSource({"ZZPRCA": again}), "ZZPRCA", tmp_path, True, TODAY)

    assert prices[0]["adj_close"] == Decimal("90.0000")
    assert prices[1]["adj_close"] == Decimal("90.5000")
    assert prices[2]["volume"] == 5


def test_an_unreadable_old_cache_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "prices" / "ZZPRCA.csv"
    path.parent.mkdir()
    path.write_text("not,a,price,file\n")

    prices, _ = cached_history(FakeSource({"ZZPRCA": SAMPLE}), "ZZPRCA", tmp_path, False, TODAY)

    assert prices == price_rows(SAMPLE, TODAY)


def test_a_damaged_cache_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "ZZPRCA.csv"
    path.write_text("trade_date,close\n2026-11-10,1\n")
    with pytest.raises(PriceError, match="columns"):
        read_csv(path, PRICE_FIELDS)

    path.write_text(",".join(PRICE_FIELDS) + "\n2026-11-10,x,1,1,1,1,5\n")
    with pytest.raises(PriceError, match="cannot be read"):
        read_csv(path, PRICE_FIELDS)


def test_split_cache_is_used_until_it_is_a_day_old(tmp_path: Path) -> None:
    history = split_series(("2015-07-15", 7.0), ("2025-11-17", 10.0))
    source = FakeSource({}, split_history={"ZZPRCA": history})

    first = cached_splits(source, "ZZPRCA", tmp_path, False)
    assert first == split_rows(history)
    assert cached_splits(source, "ZZPRCA", tmp_path, False) == first
    assert cached_splits(source, "ZZPRCB", tmp_path, False) == []  # never split: header only
    assert cached_splits(source, "ZZPRCB", tmp_path, False) == []
    assert source.split_calls == ["ZZPRCA", "ZZPRCB"]

    cached_splits(source, "ZZPRCA", tmp_path, True)
    age(tmp_path / "splits" / "ZZPRCA.csv", 2)
    cached_splits(source, "ZZPRCA", tmp_path, False)
    assert source.split_calls == ["ZZPRCA", "ZZPRCB", "ZZPRCA", "ZZPRCA"]


def test_profile_cache_is_used_and_a_failure_is_not_cached(tmp_path: Path) -> None:
    profile = {"sector": "Energy", "industry": None}
    source = FakeSource({}, {"ZZPRCA": profile})

    assert cached_profile(source, "ZZPRCA", tmp_path, False) == profile
    assert cached_profile(source, "ZZPRCA", tmp_path, False) == profile
    assert source.profile_calls == ["ZZPRCA"]
    cached_profile(source, "ZZPRCA", tmp_path, True)
    assert len(source.profile_calls) == 2

    failing = FakeSource({}, no_profile=("ZZPRCB",))
    with pytest.raises(PriceError):
        cached_profile(failing, "ZZPRCB", tmp_path, False)
    assert not (tmp_path / "profiles" / "ZZPRCB.json").exists()


# --- Database ---------------------------------------------------------------------------------

TICKERS = ("ZZPRCA", "ZZPRCB", "ZZPRCC")


@pytest.fixture
def companies(db: None) -> Any:
    """Three fixture companies, removed with their prices before and after a database test."""
    from app.db.session import admin_engine

    def wipe() -> None:
        with admin_engine.begin() as conn:
            for table in (
                "daily_prices",
                "dividends",
                "stock_splits",
                "income_statements",
                "companies",
            ):
                conn.execute(
                    text(f"DELETE FROM {table} WHERE ticker = ANY(:tickers)"),
                    {"tickers": list(TICKERS)},
                )

    wipe()
    with admin_engine.begin() as conn:
        conn.execute(
            text("INSERT INTO companies (ticker, name) VALUES (:ticker, :name)"),
            [{"ticker": ticker, "name": f"Fixture {ticker}"} for ticker in TICKERS],
        )
    yield admin_engine
    wipe()


def stored_profile(engine: Any, ticker: str) -> tuple[str | None, str | None]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT sector, industry FROM companies WHERE ticker = :ticker"),
            {"ticker": ticker},
        ).one()
    return row.sector, row.industry


def test_store_is_idempotent_and_updates_changes(companies: Any) -> None:
    prices, dividends = price_rows(SAMPLE, TODAY), dividend_rows(SAMPLE)
    splits = split_rows(split_series(("2015-07-15", 7.0), ("2025-11-17", 10.0)))
    profile: Profile = {"sector": "Technology", "industry": "Software"}

    assert store_ticker(companies, "ZZPRCA", prices, dividends, splits, profile) == 4 + 1 + 2 + 1
    assert store_ticker(companies, "ZZPRCA", prices, dividends, splits, profile) == 0

    prices[0]["adj_close"] += 1
    splits[1]["ratio"] = Decimal("0.5")
    assert store_ticker(companies, "ZZPRCA", prices, dividends, splits, profile) == 2

    with companies.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT open, high, low, close, adj_close, volume FROM daily_prices "
                "WHERE ticker = 'ZZPRCA' AND trade_date = '2026-11-10'"
            )
        ).one()
        amount = conn.execute(
            text("SELECT amount FROM dividends WHERE ticker = 'ZZPRCA' AND ex_date = '2026-11-10'")
        ).scalar_one()
        ratios = conn.execute(
            text("SELECT ratio FROM stock_splits WHERE ticker = 'ZZPRCA' ORDER BY split_date")
        ).scalars()
        assert list(ratios) == [Decimal(7), Decimal("0.5")]
    assert tuple(stored) == tuple(prices[1][name] for name in PRICE_FIELDS[1:])
    assert amount == Decimal("0.25")
    assert stored_profile(companies, "ZZPRCA") == ("Technology", "Software")


def test_prices_are_stored_in_chunks(companies: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load_prices, "UPSERT_CHUNK", 3)
    prices = price_rows(SAMPLE, TODAY)

    assert (
        store_ticker(companies, "ZZPRCA", prices, [], [], {"sector": None, "industry": None}) == 4
    )
    with companies.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM daily_prices WHERE ticker = 'ZZPRCA'")
        ).scalar_one()
    assert count == 4


def test_a_missing_sector_never_erases_a_stored_one(companies: Any) -> None:
    with companies.begin() as conn:
        assert update_profile(conn, "ZZPRCA", {"sector": "Energy", "industry": None}) == 1
        assert update_profile(conn, "ZZPRCA", {"sector": None, "industry": None}) == 0
        assert update_profile(conn, "ZZPRCA", {"sector": None, "industry": "Oil"}) == 1
        assert update_profile(conn, "ZZPRCA", {"sector": "Energy", "industry": "Oil"}) == 0

    assert stored_profile(companies, "ZZPRCA") == ("Energy", "Oil")


def test_load_runs_end_to_end_and_survives_a_bad_ticker(companies: Any, tmp_path: Path) -> None:
    source = FakeSource(
        {"ZZPRCA": SAMPLE, "ZZPRCC": frame(("2026-11-10", 50.0, 0.0), ("2026-11-11", 51.0, 0.0))},
        {"ZZPRCA": {"sector": "Technology", "industry": "Software"}},
        no_profile=("ZZPRCC",),
        split_history={"ZZPRCA": split_series(("2024-06-10", 10.0))},
    )

    first = load(companies, source, list(TICKERS), raw_dir=tmp_path, today=TODAY)
    second = load(companies, source, list(TICKERS), raw_dir=tmp_path, today=TODAY)

    assert "failed" in first["ZZPRCB"]["error"]
    assert first["ZZPRCA"] == {
        "prices": 4,
        "dividends": 1,
        "splits": 1,
        "first_date": "2026-11-09",
        "last_date": "2026-11-12",
        "changed_rows": 7,
        "missing_profile": [],
    }
    # A profile that cannot be fetched does not cost the ticker its prices.
    assert first["ZZPRCC"]["changed_rows"] == 2
    assert first["ZZPRCC"]["missing_profile"] == ["sector", "industry"]
    assert stored_profile(companies, "ZZPRCC") == (None, None)

    assert second["ZZPRCA"]["changed_rows"] == 0
    assert second["ZZPRCC"]["changed_rows"] == 0
    # The second run read prices from the cache; only the failures were asked for again.
    assert source.history_calls == ["ZZPRCA", "ZZPRCB", "ZZPRCC", "ZZPRCB"]
    assert source.split_calls == ["ZZPRCA", "ZZPRCC"]
    assert source.profile_calls == ["ZZPRCA", "ZZPRCC", "ZZPRCC"]


# --- Split-adjusted earnings per share ---------------------------------------------------------

# (fiscal_year, fiscal_quarter, period_end, filed_date, eps_diluted) -> expected factor, adjusted
ADJUSTED_CASES = [
    # Like NVDA's 10-for-1 on 2024-06-10: quarters filed before the split are divided by 10.
    ("ZZPRCA", 2024, 4, "2024-01-28", "2024-02-21", "4.9300", "10.000000", "0.4930"),
    ("ZZPRCA", 2025, 1, "2024-04-28", "2024-05-29", "5.9800", "10.000000", "0.5980"),
    ("ZZPRCA", 2025, 2, "2024-07-28", "2024-08-28", "0.6700", "1", "0.6700"),
    ("ZZPRCA", 2025, 3, "2024-10-27", "2024-11-20", None, "1", None),
    # Like WMT's 3-for-1 on 2024-02-26: the year ended before the split but the report came
    # after it, so it is already on the new basis and must not be divided again.
    ("ZZPRCB", 2024, 3, "2023-10-31", "2023-11-30", "0.1700", "3.000000", "0.0567"),
    ("ZZPRCB", 2024, 4, "2024-01-31", "2024-03-15", "0.6800", "1", "0.6800"),
    # Like NFLX: an old 7-for-1 (2015) is ignored; later splits multiply (10 x a 1-for-4 reverse).
    ("ZZPRCC", 2025, 3, "2025-09-30", "2025-10-17", "58.7000", "2.500000", "23.4800"),
    ("ZZPRCC", 2025, 4, "2025-12-31", "2026-01-23", "5.5000", "0.250000", "22.0000"),
    # No filing date known: the period end decides.
    ("ZZPRCC", 2026, 1, "2026-03-31", None, "6.0000", "0.250000", "24.0000"),
    ("ZZPRCC", 2026, 2, "2026-06-30", None, "25.0000", "1", "25.0000"),
]
ADJUSTED_SPLITS = [
    ("ZZPRCA", "2024-06-10", "10"),
    ("ZZPRCB", "2024-02-26", "3"),
    ("ZZPRCC", "2015-07-15", "7"),
    ("ZZPRCC", "2025-11-17", "10"),
    ("ZZPRCC", "2026-05-04", "0.25"),
]


@pytest.fixture
def adjusted_view(companies: Any) -> dict[tuple[str, int, int], Any]:
    """The view's rows for the fixture quarters and splits above."""
    with companies.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO income_statements (ticker, fiscal_year, fiscal_quarter, period_end, "
                "filed_date, eps_diluted, revenue) VALUES (:t, :fy, :q, :end, :filed, :eps, 100)"
            ),
            [
                {"t": t, "fy": fy, "q": q, "end": end, "filed": filed, "eps": eps}
                for t, fy, q, end, filed, eps, _, _ in ADJUSTED_CASES
            ],
        )
        conn.execute(
            text("INSERT INTO stock_splits (ticker, split_date, ratio) VALUES (:t, :day, :ratio)"),
            [{"t": t, "day": day, "ratio": ratio} for t, day, ratio in ADJUSTED_SPLITS],
        )
        rows = conn.execute(
            text("SELECT * FROM income_statements_split_adjusted WHERE ticker = ANY(:tickers)"),
            {"tickers": list(TICKERS)},
        ).all()
    return {(r.ticker, r.fiscal_year, r.fiscal_quarter): r for r in rows}


@pytest.mark.parametrize("case", ADJUSTED_CASES, ids=lambda c: f"{c[0]}-{c[1]}Q{c[2]}")
def test_eps_is_adjusted_for_the_splits_after_the_quarter_was_filed(
    adjusted_view: dict[tuple[str, int, int], Any], case: tuple[Any, ...]
) -> None:
    ticker, fiscal_year, quarter, _, _, eps, factor, adjusted = case
    row = adjusted_view[(ticker, fiscal_year, quarter)]

    assert row.split_factor == Decimal(factor)
    assert row.eps_diluted_adjusted == (Decimal(adjusted) if adjusted else None)
    # The reported figure and the other columns pass through unchanged.
    assert row.eps_diluted == (Decimal(eps) if eps else None)
    assert row.revenue == 100


def test_the_adjusted_view_has_one_row_per_quarter(
    adjusted_view: dict[tuple[str, int, int], Any],
) -> None:
    assert len(adjusted_view) == len(ADJUSTED_CASES)


def test_main_rejects_a_ticker_that_is_not_loaded(db: None, yahoo: type[FakeTicker]) -> None:
    assert load_prices.main(["--tickers", "ZZNOPE"]) == 1
    assert yahoo.symbols == []


# --- load_all ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "sec_code", "prices_code", "expected", "prices_ran"),
    [
        ([], 0, 0, 0, True),
        (["--tickers", "AAPL,MSFT", "--refresh"], 0, 0, 0, True),
        ([], 0, 1, 1, True),
        (["--refresh"], 1, 0, 1, False),
    ],
)
def test_load_all_runs_sec_then_prices(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    sec_code: int,
    prices_code: int,
    expected: int,
    prices_ran: bool,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake(name: str, code: int) -> Any:
        def run(argv: list[str]) -> int:
            calls.append((name, argv))
            return code

        return run

    monkeypatch.setattr(load_sec, "main", fake("sec", sec_code))
    monkeypatch.setattr(load_prices, "main", fake("prices", prices_code))

    assert load_all.main(arguments) == expected
    steps = [("sec", arguments), ("prices", arguments)]
    assert calls == (steps if prices_ran else steps[:1])
