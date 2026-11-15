"""Tests for the retrieve_schema node: retrieval and the schema are replaced, so no database."""

from typing import Any

import pytest

from app.agent.nodes import retrieve_schema as node_module
from app.agent.nodes.retrieve_schema import format_table, full_schema, retrieve_schema
from app.db.retrieval import TableDocs

SCHEMA = {
    "companies": (("ticker", "text"), ("name", "text")),
    "income_statements": (
        ("ticker", "text"),
        ("fiscal_year", "smallint"),
        ("revenue", "numeric"),
        ("gross_profit", "numeric"),
    ),
    "income_statements_split_adjusted": (
        ("ticker", "text"),
        ("fiscal_year", "smallint"),
        ("eps_diluted", "numeric"),
        ("eps_diluted_adjusted", "numeric"),
    ),
}
DOCS = {
    "income_statements": {
        None: "Quarterly income statement per company.",
        "ticker": "Company ticker.",
        "revenue": "Total revenue for the quarter in USD.",
        "gross_profit": "Revenue minus cost of revenue.",
        "ceo": "A column that no longer exists.",
    },
    "companies": {None: "One row per company.", "ticker": "Stock ticker symbol."},
    "income_statements_split_adjusted": {
        None: "View: income_statements plus split-adjusted EPS.",
        "eps_diluted_adjusted": "Diluted EPS on today's share basis.",
    },
}
INCOME = TableDocs("income_statements", "x", {"gross_profit": "x"}, 0.2)
COMPANIES = TableDocs("companies", "x", {"ticker": "x"})
EXAMPLE = {"question": "What was Amazon's gross margin?", "sql": "SELECT 1"}


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the schema and the searches; the test sets what they return."""
    found: dict[str, Any] = {"tables": [INCOME, COMPANIES], "examples": [EXAMPLE], "asked": []}

    def search_schema(question: str, k: int) -> list[TableDocs]:
        found["asked"].append(("schema", question, k))
        return found["tables"]

    def search_examples(question: str, k: int) -> list[dict[str, str]]:
        found["asked"].append(("examples", question, k))
        return found["examples"]

    def table_docs(tables: list[str]) -> dict[str, dict[str | None, str]]:
        found["asked"].append(("docs", tuple(tables)))
        return {table: DOCS.get(table, {}) for table in tables}

    monkeypatch.setattr(node_module.reference, "schema_columns", lambda: SCHEMA)
    monkeypatch.setattr(node_module.retrieval, "search_schema", search_schema)
    monkeypatch.setattr(node_module.retrieval, "search_examples", search_examples)
    monkeypatch.setattr(node_module.retrieval, "table_docs", table_docs)
    return found


def test_each_chosen_table_is_shown_whole_with_types_and_descriptions(
    searches: dict[str, Any],
) -> None:
    update = retrieve_schema({"question": "Apple gross margin trend"})

    assert set(update) == {"schema_context", "examples", "trace"}
    # Only gross_profit matched the search, but revenue is needed for a margin and is shown too.
    assert update["schema_context"] == (
        "-- income_statements: Quarterly income statement per company.\n"
        "income_statements (\n"
        "  ticker text,  -- Company ticker.\n"
        "  fiscal_year smallint,\n"
        "  revenue numeric,  -- Total revenue for the quarter in USD.\n"
        "  gross_profit numeric,  -- Revenue minus cost of revenue.\n"
        ")\n"
        "\n"
        "-- companies: One row per company.\n"
        "companies (\n"
        "  ticker text,  -- Stock ticker symbol.\n"
        "  name text,\n"
        ")"
    )
    assert update["examples"] == [EXAMPLE]
    assert update["trace"][0]["note"] == "tables: income_statements, companies; 1 examples"
    assert searches["asked"] == [
        ("schema", "Apple gross margin trend", 12),
        ("docs", ("income_statements", "companies")),
        ("examples", "Apple gross margin trend", 3),
    ]


def test_a_described_column_that_no_longer_exists_is_left_out(searches: dict[str, Any]) -> None:
    assert "ceo" not in retrieve_schema({"question": "who runs Apple"})["schema_context"]


def test_a_view_shows_its_undescribed_columns_too(searches: dict[str, Any]) -> None:
    searches["tables"] = [TableDocs("income_statements_split_adjusted", "x", {"eps": "x"})]

    context = retrieve_schema({"question": "Nvidia EPS trend"})["schema_context"]

    assert "  ticker text,\n  fiscal_year smallint,\n  eps_diluted numeric,\n" in context
    assert "  eps_diluted_adjusted numeric,  -- Diluted EPS on today's share basis.\n" in context


def test_a_retrieved_table_that_is_not_queryable_is_dropped(searches: dict[str, Any]) -> None:
    searches["tables"] = [TableDocs("old_table", "Dropped long ago.", {"x": "y"}), COMPANIES]

    update = retrieve_schema({"question": "anything"})

    assert "old_table" not in update["schema_context"]
    assert update["trace"][0]["note"].startswith("tables: companies;")


@pytest.mark.parametrize("tables", [[], [TableDocs("companies")]])
def test_without_schema_docs_the_whole_schema_is_the_context(
    searches: dict[str, Any], tables: list[TableDocs]
) -> None:
    searches["tables"] = tables

    update = retrieve_schema({"question": "Apple revenue"})

    assert update["schema_context"] == full_schema(SCHEMA)
    assert "  revenue numeric,\n" in update["schema_context"]
    assert "income_statements_split_adjusted (" in update["schema_context"]
    assert update["examples"] == []
    assert update["trace"][0]["note"] == "no schema docs found; using the full schema"
    assert [entry[0] for entry in searches["asked"]] == ["schema"]  # nothing else is searched


def test_no_examples_is_fine(searches: dict[str, Any]) -> None:
    searches["examples"] = []

    update = retrieve_schema({"question": "Apple gross margin trend"})

    assert update["examples"] == []
    assert update["trace"][0]["note"].endswith("0 examples")


def test_format_table_without_any_description() -> None:
    assert format_table("companies", SCHEMA["companies"], {}) == (
        "-- companies\ncompanies (\n  ticker text,\n  name text,\n)"
    )
