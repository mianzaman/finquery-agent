"""Tests for the retrieve_schema node: retrieval and the schema are replaced, so no database."""

from typing import Any

import pytest

from app.agent.nodes import retrieve_schema as node_module
from app.agent.nodes.retrieve_schema import format_table, full_schema, retrieve_schema
from app.db.retrieval import TableDocs

SCHEMA = {
    "companies": (("ticker", "text"), ("name", "text"), ("sector", "text")),
    "income_statements": (
        ("ticker", "text"),
        ("fiscal_year", "smallint"),
        ("fiscal_quarter", "smallint"),
        ("period_end", "date"),
        ("revenue", "numeric"),
        ("gross_profit", "numeric"),
        ("net_income", "numeric"),
    ),
    "income_statements_split_adjusted": (
        ("ticker", "text"),
        ("fiscal_year", "smallint"),
        ("fiscal_quarter", "smallint"),
        ("period_end", "date"),
        ("eps_diluted", "numeric"),
        ("eps_diluted_adjusted", "numeric"),
    ),
}
INCOME = TableDocs(
    "income_statements",
    "Quarterly income statement per company.",
    {"gross_profit": "Revenue minus cost of revenue.", "ticker": "Company ticker."},
    0.2,
)
COMPANIES = TableDocs("companies", "One row per company.", {"ticker": "Stock ticker symbol."})
EXAMPLE = {"question": "What was Amazon's gross margin?", "sql": "SELECT 1"}


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the schema and both searches; the test sets what they return."""
    found: dict[str, Any] = {"tables": [INCOME, COMPANIES], "examples": [EXAMPLE], "asked": []}

    def search_schema(question: str, k: int) -> list[TableDocs]:
        found["asked"].append(("schema", question, k))
        return found["tables"]

    def search_examples(question: str, k: int) -> list[dict[str, str]]:
        found["asked"].append(("examples", question, k))
        return found["examples"]

    monkeypatch.setattr(node_module.reference, "schema_columns", lambda: SCHEMA)
    monkeypatch.setattr(node_module.retrieval, "search_schema", search_schema)
    monkeypatch.setattr(node_module.retrieval, "search_examples", search_examples)
    return found


def test_the_context_shows_the_chosen_tables_with_types_and_descriptions(
    searches: dict[str, Any],
) -> None:
    update = retrieve_schema({"question": "Apple gross margin trend"})

    assert set(update) == {"schema_context", "examples", "trace"}
    assert update["schema_context"] == (
        "-- income_statements: Quarterly income statement per company.\n"
        "income_statements (\n"
        "  ticker text,  -- Company ticker.\n"
        "  fiscal_year smallint,\n"
        "  fiscal_quarter smallint,\n"
        "  period_end date,\n"
        "  gross_profit numeric,  -- Revenue minus cost of revenue.\n"
        ")\n"
        "\n"
        "-- companies: One row per company.\n"
        "companies (\n"
        "  ticker text,  -- Stock ticker symbol.\n"
        ")"
    )
    assert update["examples"] == [EXAMPLE]
    assert update["trace"][0]["note"] == "tables: income_statements, companies; 1 examples"
    assert searches["asked"] == [
        ("schema", "Apple gross margin trend", 12),
        ("examples", "Apple gross margin trend", 3),
    ]


def test_key_columns_appear_even_when_nobody_described_them(searches: dict[str, Any]) -> None:
    # The owner's docs describe only the two extra columns of the view.
    view = TableDocs(
        "income_statements_split_adjusted",
        "View: income_statements plus split-adjusted EPS.",
        {"eps_diluted_adjusted": "Diluted EPS on today's share basis."},
    )
    searches["tables"] = [view]

    context = retrieve_schema({"question": "Nvidia EPS trend"})["schema_context"]

    assert "  ticker text,\n  fiscal_year smallint,\n  fiscal_quarter smallint,\n" in context
    assert "  eps_diluted_adjusted numeric,  -- Diluted EPS on today's share basis.\n" in context
    assert "  eps_diluted numeric" not in context  # not retrieved and not a key


def test_a_described_table_or_column_that_no_longer_exists_is_left_out(
    searches: dict[str, Any],
) -> None:
    gone = TableDocs("old_table", "Dropped long ago.", {"x": "Something."})
    stale = TableDocs("companies", "One row per company.", {"ticker": "Ticker.", "ceo": "Gone."})
    searches["tables"] = [gone, stale]

    context = retrieve_schema({"question": "who runs Apple"})["schema_context"]

    assert "old_table" not in context and "ceo" not in context
    assert context.startswith("-- companies: One row per company.")


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
    assert [kind for kind, _, _ in searches["asked"]] == ["schema"]  # no example search either


def test_no_examples_is_fine(searches: dict[str, Any]) -> None:
    searches["examples"] = []

    update = retrieve_schema({"question": "Apple gross margin trend"})

    assert update["examples"] == []
    assert update["trace"][0]["note"].endswith("0 examples")


def test_format_table_without_a_description() -> None:
    text = format_table(TableDocs("companies", "", {"name": "Company name."}), SCHEMA["companies"])

    assert text == "-- companies\ncompanies (\n  ticker text,\n  name text,  -- Company name.\n)"
