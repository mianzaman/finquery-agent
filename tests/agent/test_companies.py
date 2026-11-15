"""Tests for matching company names to tickers. Pure functions on fixture companies."""

import pytest

from app.agent.companies import (
    build_index,
    describe,
    is_generic,
    is_sector,
    load_aliases,
    normalize,
    resolve_mention,
    scan_question,
)
from app.db.reference import Company
from tests.agent.fixtures import ALIASES, COMPANIES

INDEX = build_index(COMPANIES, ALIASES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Apple", "apple"),
        ("Apple's", "apple"),
        ("APPLE INC.", "apple"),
        ("The Home Depot", "home depot"),
        ("Alphabet Inc", "alphabet"),
        ("Southern Company", "southern"),
        ("Nvidia stock", "nvidia"),
        ("Coca-Cola", "coca cola"),
        ("Procter & Gamble", "procter & gamble"),
        ("  Meta   Platforms  ", "meta platforms"),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("mention", "ticker"),
    [
        ("Apple", "AAPL"),
        ("apple's", "AAPL"),
        ("Google", "GOOGL"),  # alias
        ("Alphabet Inc.", "GOOGL"),
        ("JP Morgan", "JPM"),
        ("MSFT", "MSFT"),  # ticker
        ("msft", "MSFT"),
        ("$GS", "GS"),
        ("V", "V"),  # a one-letter ticker counts when written in capitals
        ("Microsfot", "MSFT"),  # misspelled
        ("Microsoft Corporation", "MSFT"),
        ("Wallmart", "WMT"),
        ("Facebok", "META"),
        ("General Electric", "GE"),
    ],
)
def test_a_known_company_resolves_to_its_ticker(mention: str, ticker: str) -> None:
    assert resolve_mention(mention, INDEX) == (ticker, [])


@pytest.mark.parametrize(
    ("mention", "suggestions"),
    [
        ("Samsung", []),
        ("Toyota", []),
        ("v", []),  # lower-case single letter: not a ticker
        ("Gs", []),  # too short to guess
        ("Goldman Stanley", ["GS"]),
        ("Micro Strategy", ["MSFT"]),
    ],
)
def test_an_unknown_company_gives_close_matches_only(mention: str, suggestions: list[str]) -> None:
    assert resolve_mention(mention, INDEX) == (None, suggestions)


@pytest.mark.parametrize(
    ("question", "tickers"),
    [
        ("Compare Visa's and Microsoft's net income in fiscal 2025.", ["V", "MSFT"]),
        ("Google vs Microsoft 2023", ["GOOGL", "MSFT"]),
        ("What were JPMorgan Chase's total assets?", ["JPM"]),  # one company, not JPM twice
        ("How did Home Depot do against WMT and $GE?", ["HD", "WMT", "GE"]),
        ("WHAT WAS APPLE REVENUE", ["AAPL"]),
        ("So what did the cat do in a V-shaped recovery? ge is fine", []),
        ("Which Utilities companies had the most debt?", []),
        ("Rank the 5 largest companies by revenue", []),
        ("What did Applebee's earn?", []),  # "apple" inside another word
    ],
)
def test_scan_question_finds_names_aliases_and_capital_tickers(
    question: str, tickers: list[str]
) -> None:
    assert scan_question(question, INDEX) == tickers


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("Technology", True),
        ("Utilities companies", True),
        ("the Financial Services sector", True),
        ("Apple", False),
        ("Southern Company", False),
    ],
)
def test_a_sector_is_not_a_company(mention: str, expected: bool) -> None:
    assert is_sector(mention, INDEX) is expected


def test_a_name_shared_by_two_companies_matches_neither() -> None:
    twins = (Company("AAA", "Acme", None, 12), Company("BBB", "Beta", None, 12))
    index = build_index(twins, {"AAA": ["Shared Name"], "BBB": ["Shared Name"]})

    assert resolve_mention("Shared Name", index)[0] is None
    assert resolve_mention("Acme", index) == ("AAA", [])


def test_describe_names_the_suggestions() -> None:
    assert describe(["MSFT", "META"], INDEX) == "Microsoft (MSFT), Meta Platforms (META)"


def test_the_alias_file_only_names_real_tickers_and_no_alias_twice() -> None:
    from ingestion.load_sec import read_companies

    aliases = load_aliases()
    tickers = {ticker for ticker, _ in read_companies()}
    assert set(aliases) <= tickers
    assert "Google" in aliases["GOOGL"]
    seen: dict[str, str] = {}
    for ticker, names in aliases.items():
        assert isinstance(names, list) and all(isinstance(name, str) for name in names)
        for name in names:
            assert seen.setdefault(normalize(name), ticker) == ticker, f"{name} is listed twice"


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("S&P 500", True),
        ("the S&P 500", True),
        ("Nasdaq", True),
        ("the market", True),
        ("AI", True),
        ("Apple", False),
        ("Samsung", False),  # an unknown company is still refused, not ignored
        ("Meta", False),
    ],
)
def test_an_index_or_a_generic_word_is_not_a_company(mention: str, expected: bool) -> None:
    assert is_generic(mention) is expected
