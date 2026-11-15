"""Turns company names in a question into tickers, without a model.

A name matches when it equals a company's name, ticker or alias (ignoring case, punctuation and
endings like "Inc"), or is a near miss of one ("Microsfot"). Pure functions: the company list
and the aliases are passed in, so nothing here touches the database.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from app.db.reference import Company

ALIASES_PATH = Path(__file__).resolve().parent.parent.parent / "ingestion" / "company_aliases.yaml"

MATCH_SCORE = 85  # at or above: the same company, misspelled
SUGGEST_SCORE = 60  # at or above: worth offering as "did you mean"
MAX_SUGGESTIONS = 3
MIN_FUZZY_LENGTH = 4  # shorter names are too easy to confuse ("GE" with "GS")

_ENDINGS = re.compile(
    r"\s+(?:inc|incorporated|corp|corporation|co|company|ltd|plc|llc|holdings|group|"
    r"stock|shares|share price|stock price)$"
)
_POSSESSIVE = re.compile(r"['’]s\b")
_NOT_NAME = re.compile(r"[^a-z0-9& ]+")


def normalize(text: str) -> str:
    """Lower-case a name and drop possessives, punctuation, a leading "the" and legal endings."""
    cleaned = _POSSESSIVE.sub("", text.lower().replace("-", " "))
    cleaned = " ".join(_NOT_NAME.sub("", cleaned).split())
    cleaned = cleaned.removeprefix("the ")
    while cleaned != (shorter := _ENDINGS.sub("", cleaned)):
        cleaned = shorter
    return cleaned


@dataclass
class CompanyIndex:
    """Every way a company can be written, pointing at its ticker."""

    keys: dict[str, str] = field(default_factory=dict)  # normalized name or alias -> ticker
    tickers: dict[str, str] = field(default_factory=dict)  # ticker -> company name
    sectors: frozenset[str] = frozenset()  # normalized sector names, which are not companies


def build_index(companies: tuple[Company, ...], aliases: dict[str, list[str]]) -> CompanyIndex:
    """Index names and aliases. A key claimed by two companies is dropped as ambiguous."""
    index = CompanyIndex(
        tickers={c.ticker: c.name for c in companies},
        sectors=frozenset(normalize(c.sector) for c in companies if c.sector),
    )
    ambiguous: set[str] = set()
    for company in companies:
        for name in [company.name, *aliases.get(company.ticker, [])]:
            key = normalize(str(name))
            if not key:
                continue
            if index.keys.get(key, company.ticker) != company.ticker:
                ambiguous.add(key)
            index.keys[key] = company.ticker
    for key in ambiguous:
        del index.keys[key]
    return index


@lru_cache
def load_aliases(path: Path = ALIASES_PATH) -> dict[str, list[str]]:
    """Read the alias file: `{ticker: [other names]}`."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# Things a model may list as a "company" that are not one. They are skipped, not refused.
NOT_COMPANIES = frozenset(
    {
        "s&p 500", "s&p", "s&p500", "sp500", "sp 500", "dow", "dow jones", "djia", "nasdaq",
        "nasdaq 100", "russell 2000", "market", "stock market", "index", "wall street",
        "tech", "big tech", "ai", "faang", "magnificent seven", "magnificent 7", "sec",
        "companies", "company", "sector", "sectors", "industry", "peers", "competitors",
    }
)  # fmt: skip


def is_generic(mention: str) -> bool:
    """True for an index, a market or a generic word such as "S&P 500" or "the market"."""
    return normalize(mention) in NOT_COMPANIES


def is_sector(mention: str, index: CompanyIndex) -> bool:
    """True for "Technology" or "Utilities companies": a group of companies, not one."""
    key = normalize(mention)
    key = re.sub(r"\s+(?:companies|sector|stocks|firms|industry)$", "", key)
    return key in index.sectors


def resolve_mention(mention: str, index: CompanyIndex) -> tuple[str | None, list[str]]:
    """Return `(ticker, [])` for a recognised company, else `(None, close tickers)`."""
    written = mention.strip().lstrip("$")
    key = normalize(written)
    if written.upper() in index.tickers and (written.isupper() or len(written) >= 3):
        return written.upper(), []
    if key in index.keys:
        return index.keys[key], []
    if len(key) < MIN_FUZZY_LENGTH:
        return None, []
    scored = process.extract(key, list(index.keys), scorer=fuzz.ratio, limit=10)
    if scored and scored[0][1] >= MATCH_SCORE:
        return index.keys[scored[0][0]], []
    suggestions: list[str] = []
    for candidate, score, _ in scored:
        ticker = index.keys[candidate]
        if score >= SUGGEST_SCORE and ticker not in suggestions:
            suggestions.append(ticker)
    return None, suggestions[:MAX_SUGGESTIONS]


def scan_question(question: str, index: CompanyIndex) -> list[str]:
    """Tickers of the companies written out in the question, in order of appearance.

    Names and aliases match as whole words in any case. A ticker must be in capitals and at
    least two letters long, so "so", "cat" or "a" in a sentence never count.
    """
    found: dict[int, str] = {}
    text = " ".join(_NOT_NAME.sub(" ", _POSSESSIVE.sub("", question.lower())).split())
    taken: list[range] = []
    for key in sorted(index.keys, key=len, reverse=True):  # "home depot" before a shorter key
        for match in re.finditer(rf"(?<![a-z0-9&]){re.escape(key)}(?![a-z0-9&])", text):
            span = range(match.start(), match.end())
            if not any(span.start < other.stop and other.start < span.stop for other in taken):
                taken.append(span)
                # Positions in the lowered text and in the question differ; order is what counts.
                found[match.start()] = index.keys[key]
    ordered = [found[position] for position in sorted(found)]
    if not question.isupper():
        for match in re.finditer(r"(?<![A-Za-z0-9&])\$?([A-Z]{2,5})(?![A-Za-z0-9&])", question):
            if match.group(1) in index.tickers:
                ordered.append(match.group(1))
    return list(dict.fromkeys(ordered))


def describe(tickers: list[str], index: CompanyIndex) -> str:
    """ "Microsoft (MSFT), Meta Platforms (META)" for a refusal message."""
    return ", ".join(f"{index.tickers[t]} ({t})" for t in tickers)
