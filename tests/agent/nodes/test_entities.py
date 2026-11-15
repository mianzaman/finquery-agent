"""Tests for the entities node: fixture companies and calendars, no database and no model."""

from typing import Any

import pytest

from app.agent.companies import build_index
from app.agent.nodes import entities as entities_module
from app.agent.nodes.entities import entities, resolve_companies, unknown_company_message
from tests.agent.fixtures import ALIASES, CALENDAR, COMPANIES

INDEX = build_index(COMPANIES, ALIASES)


@pytest.fixture(autouse=True)
def reference_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the fixture companies and calendars instead of reading the database."""
    monkeypatch.setattr(entities_module.reference, "companies", lambda: COMPANIES)
    monkeypatch.setattr(entities_module.reference, "fiscal_calendar", lambda: CALENDAR)
    monkeypatch.setattr(entities_module, "load_aliases", lambda: ALIASES)


def periods_of(update: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [(p["ticker"], p["fiscal_year"], p["fiscal_quarter"]) for p in update["periods"]]


def test_apple_last_quarter_is_apples_own_fiscal_quarter() -> None:
    update = entities({"question": "Apple last quarter", "company_mentions": ["Apple"]})

    assert set(update) == {"tickers", "periods", "trace"}  # no refusal keys when all is well
    assert update["tickers"] == ["AAPL"]
    # Apple's fiscal year ends in September: the quarter to 30 September is fiscal Q4.
    assert periods_of(update) == [("AAPL", 2026, 4)]
    assert update["periods"][0]["period_end"] == "2026-09-30"
    assert update["trace"][0]["note"] == "1 companies, 1 periods"


def test_google_vs_microsoft_2023_gives_each_its_own_fiscal_year() -> None:
    state = {"question": "Google vs Microsoft 2023", "company_mentions": ["Google", "Microsoft"]}

    update = entities(state)

    assert update["tickers"] == ["GOOGL", "MSFT"]
    assert periods_of(update) == [("GOOGL", 2023, q) for q in (1, 2, 3, 4)] + [
        ("MSFT", 2023, q) for q in (1, 2, 3, 4)
    ]
    ends = {p["ticker"]: p["period_end"] for p in update["periods"]}
    assert ends == {"GOOGL": "2023-12-31", "MSFT": "2023-06-30"}  # the last quarter of each


def test_a_misspelled_company_is_still_found() -> None:
    state = {"question": "Microsfot revenue in fiscal 2025", "company_mentions": ["Microsfot"]}

    update = entities(state)

    assert update["tickers"] == ["MSFT"]
    assert "refused" not in update


def test_an_unknown_company_is_refused_with_suggestions() -> None:
    state = {"question": "Goldman Stanley revenue in 2025", "company_mentions": ["Goldman Stanley"]}

    update = entities(state)

    assert update["refused"] is True
    assert update["refusal_reason"] == (
        'I could not find "Goldman Stanley" among the companies I have data for. '
        "Did you mean Goldman Sachs (GS)? The data covers 12 large US companies."
    )
    assert (update["tickers"], update["periods"]) == ([], [])
    assert update["trace"][0]["note"] == "unknown company: Goldman Stanley"


def test_an_unknown_company_with_nothing_close_is_refused_plainly() -> None:
    update = entities({"question": "Samsung's revenue in 2025", "company_mentions": ["Samsung"]})

    assert update["refused"] is True
    assert update["refusal_reason"] == (
        'I could not find "Samsung" among the companies I have data for. '
        "The data covers 12 large US companies."
    )


def test_one_unknown_company_refuses_even_when_another_is_known() -> None:
    state = {"question": "Compare Apple and Samsung", "company_mentions": ["Apple", "Samsung"]}

    update = entities(state)

    assert update["refused"] is True
    assert update["tickers"] == ["AAPL"]
    assert '"Samsung"' in update["refusal_reason"] and '"Apple"' not in update["refusal_reason"]


@pytest.mark.parametrize(
    ("question", "mentions"),
    [
        ("Which 5 companies had the highest revenue in fiscal 2025?", []),
        ("Rank the Utilities companies by total assets last quarter", ["Utilities companies"]),
        ("Which Technology company had the highest margin?", ["Technology"]),
    ],
)
def test_a_question_that_names_no_company_is_not_refused(
    question: str, mentions: list[str]
) -> None:
    update = entities({"question": question, "company_mentions": mentions})

    assert "refused" not in update
    assert update["tickers"] == []
    assert all(period["ticker"] is None for period in update["periods"])


@pytest.mark.parametrize(
    "mentions",
    [["S&P 500"], ["the S&P 500", "Apple"], ["Apple", "the market", "Nasdaq", "AI", "Big Tech"]],
)
def test_an_index_or_market_listed_as_a_company_is_ignored(mentions: list[str]) -> None:
    question = "Did Apple beat the S&P 500 in fiscal 2025?"

    update = entities({"question": question, "company_mentions": mentions})

    assert "refused" not in update
    assert update["tickers"] == ["AAPL"]
    assert periods_of(update) == [("AAPL", 2025, q) for q in (1, 2, 3, 4)]


def test_the_model_listing_the_sp_500_does_not_refuse_the_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole front of the graph with a fake model: guardrail's reply feeds entities.
    from datetime import date

    from pydantic import BaseModel

    from app import llm
    from app.agent.nodes import guardrail as guardrail_module
    from tests.fakes import ScriptedChat, reply

    verdict = '{"allowed": true, "reason": "", "intent": "comparison", "companies": ["S&P 500"]}'
    fake = ScriptedChat(replies=[reply(verdict)])

    def structured(schema: type[BaseModel], messages: list) -> BaseModel:
        return llm.structured(schema, messages, model=fake, native=False)

    monkeypatch.setattr(guardrail_module, "structured", structured)
    monkeypatch.setattr(
        guardrail_module.reference, "data_range", lambda: (date(2021, 1, 2), date(2026, 11, 13))
    )
    state: dict[str, Any] = {"question": "Which companies beat the S&P 500 in calendar 2025?"}

    state |= guardrail_module.guardrail(state)
    update = entities(state)

    assert state["company_mentions"] == ["S&P 500"]
    assert "refused" not in update
    assert update["tickers"] == []
    assert [(p["ticker"], p["basis"], p["period_end"]) for p in update["periods"]] == [
        (None, "calendar", "2025-12-31")
    ]


def test_a_company_the_model_did_not_list_is_found_by_the_scan() -> None:
    update = entities({"question": "Compare Visa and JP Morgan in FY25", "company_mentions": []})

    assert update["tickers"] == ["V", "JPM"]
    assert {p["ticker"] for p in update["periods"]} == {"V", "JPM"}


@pytest.mark.parametrize("mentions", [None, [], [None, 5, "", "  ", ["Walmart"]]])
def test_missing_or_malformed_mentions_are_tolerated(mentions: Any) -> None:
    state = {"question": "Walmart revenue YTD", "company_mentions": mentions}

    update = entities(state)

    assert update["tickers"] == ["WMT"]
    assert "refused" not in update
    assert entities({"question": "Walmart revenue YTD"})["tickers"] == ["WMT"]


def test_an_unknown_name_is_quoted_back_short_and_on_one_line() -> None:
    hostile = 'Foo{0.__class__}\n<script>"x"' + "y" * 200

    reason = entities({"question": "revenue?", "company_mentions": [hostile]})["refusal_reason"]

    assert "\n" not in reason
    assert reason.startswith("I could not find \"Foo{0.__class__} <script>'x'yyy")
    assert len(reason) < 200


def test_a_company_named_twice_is_listed_once_in_question_order() -> None:
    mentions = ["Microsoft", "Google", "MSFT"]

    tickers, unknown = resolve_companies("Microsoft vs Google: is MSFT ahead?", mentions, INDEX)

    assert (tickers, unknown) == (["MSFT", "GOOGL"], {})


def test_a_period_outside_the_data_is_noted_in_the_trace() -> None:
    update = entities({"question": "Apple revenue in fiscal 2019", "company_mentions": ["Apple"]})

    assert update["periods"] == []
    assert update["trace"][0]["note"] == "1 companies, 0 periods; no data for AAPL fiscal 2019"


def test_unknown_company_message_lists_every_unknown_name() -> None:
    message = unknown_company_message({"Samsung": [], "Micro Strategy": ["MSFT"]}, INDEX)

    assert message == (
        'I could not find "Samsung" among the companies I have data for. '
        'I could not find "Micro Strategy" among the companies I have data for. '
        "Did you mean Microsoft (MSFT)? The data covers 12 large US companies."
    )
