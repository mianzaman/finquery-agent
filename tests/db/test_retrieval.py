"""Retrieval tests: fake vectors, and a scratch schema so the real tables are untouched."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from app.db import retrieval
from app.db.retrieval import RetrievalError, search_examples, search_schema, vector_literal
from ingestion import build_embeddings
from ingestion.build_embeddings import (
    BuildError,
    Example,
    SchemaDoc,
    build,
    read_examples,
    read_schema_docs,
)
from tests.fake_embeddings import FakeEmbeddings, embed

FIXTURES = Path(__file__).parent.parent / "fixtures"
DOCS = read_schema_docs(FIXTURES / "schema_docs_sample.yaml")
EXAMPLES = read_examples(FIXTURES / "examples_sample.jsonl")
SCRATCH = "test_retrieval"
MODEL = "fake:bag-of-words"


@pytest.fixture
def scratch(db: None, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Empty copies of the two retrieval tables in their own schema, dropped afterwards."""
    from app.db.session import admin_engine

    with admin_engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCRATCH} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCRATCH}"))
        for table in ("schema_docs", "query_examples"):
            conn.execute(
                text(f"CREATE TABLE {SCRATCH}.{table} (LIKE public.{table} INCLUDING ALL)")
            )
        conn.execute(text(f"GRANT USAGE ON SCHEMA {SCRATCH} TO agent_reader"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCRATCH} TO agent_reader"))
    monkeypatch.setattr(retrieval, "RETRIEVAL_SCHEMA", SCRATCH)
    fake = FakeEmbeddings()
    monkeypatch.setattr(retrieval, "get_embeddings", lambda: fake)
    retrieval.embed_question.cache_clear()
    yield admin_engine, fake
    retrieval.embed_question.cache_clear()
    with admin_engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCRATCH} CASCADE"))


@pytest.fixture
def built(scratch: Any) -> Any:
    """The scratch tables filled from the fixture files."""
    engine, fake = scratch
    build(engine, fake.embed_documents, MODEL, DOCS, EXAMPLES)
    return engine, fake


def rows(engine: Any, table: str, columns: str = "*") -> list[Any]:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT {columns} FROM {SCRATCH}.{table} ORDER BY 1, 2")).all()


# --- Searching --------------------------------------------------------------------------------


def test_apple_gross_margin_trend_retrieves_income_statements(built: Any) -> None:
    tables = search_schema("Apple gross margin trend", k=3)

    assert tables[0].table == "income_statements"
    assert "gross_profit" in tables[0].columns
    assert tables[0].description.startswith("Quarterly income statement per company")


def test_share_price_retrieves_daily_prices(built: Any) -> None:
    tables = search_schema("share price", k=3)

    assert tables[0].table == "daily_prices"
    assert "close" in tables[0].columns


def test_a_retrieved_table_always_comes_with_its_key_columns(built: Any) -> None:
    (income,) = [t for t in search_schema("gross margin", k=1) if t.table == "income_statements"]

    # Only gross_profit matched, but a query cannot be written without the keys.
    assert set(income.columns) == {
        "gross_profit",
        "ticker",
        "fiscal_year",
        "fiscal_quarter",
        "period_end",
    }
    assert "net_income" not in income.columns and "revenue" not in income.columns


def test_companies_is_always_included_and_comes_after_the_matches(built: Any) -> None:
    tables = search_schema("dividend paid per share", k=2)

    assert [t.table for t in tables] == ["dividends", "companies"]
    companies = tables[-1]
    assert companies.description == "One row per company with its ticker, name and sector."
    assert set(companies.columns) == {"ticker"}  # its key column; nothing else matched
    assert [t.table for t in search_schema("dividend paid per share", k=2, always=())] == [
        "dividends"
    ]


def test_a_table_description_is_a_row_of_its_own_and_can_match(built: Any) -> None:
    engine, _ = built

    stored = rows(engine, "schema_docs", "table_name, column_name")
    assert ("daily_prices", None) in [tuple(row) for row in stored]
    # "bars" appears only in the table description of daily_prices.
    assert search_schema("daily bars from Yahoo Finance", k=1, always=())[0].table == "daily_prices"


def test_examples_are_returned_closest_first(built: Any) -> None:
    examples = search_examples("What was Nvidia's closing share price on 3 March 2025?", k=2)

    assert examples[0] == {
        "question": "What was Tesla's closing share price on 2 January 2025?",
        "sql": "SELECT close FROM daily_prices WHERE ticker = 'TSLA' AND trade_date = '2025-01-02'",
    }
    assert len(examples) == 2
    assert len(search_examples("dividends per share", k=10)) == 4  # never more than exist


def test_both_searches_embed_the_question_once(built: Any) -> None:
    _, fake = built

    search_schema("gross margin trend")
    search_examples("gross margin trend")
    search_schema("gross margin trend")

    assert fake.queries == ["gross margin trend"]


def test_empty_tables_return_nothing(scratch: Any) -> None:
    assert search_schema("share price", always=()) == []
    assert search_examples("share price") == []
    # With nothing built, the always-included table has no description or columns.
    (companies,) = search_schema("share price")
    assert (companies.table, companies.description, companies.columns) == ("companies", "", {})


def test_a_question_with_sql_in_it_is_only_ever_a_value(built: Any) -> None:
    engine, _ = built
    hostile = "share price'; DROP TABLE test_retrieval.schema_docs; -- %s %(k)s"

    assert search_schema(hostile, k=2)  # it searches; the text is bound, never executed
    assert len(rows(engine, "schema_docs", "table_name, column_name")) == len(DOCS)


def test_a_vector_of_the_wrong_size_is_refused() -> None:
    assert vector_literal([0.5] * 384).startswith("[0.5,0.5,")
    with pytest.raises(RetrievalError, match="1536 dimensions, not 384"):
        vector_literal([0.1] * 1536)


def test_the_reader_role_can_search_the_real_tables_but_not_write_them(db: None) -> None:
    from app.db.session import QueryError, run_select

    assert run_select("SELECT count(*) FROM schema_docs")[1][0][0] >= 0
    assert run_select("SELECT count(*) FROM query_examples")[1][0][0] >= 0
    with pytest.raises(QueryError, match="read-only transaction|permission denied"):
        run_select("DELETE FROM schema_docs")


# --- Building ---------------------------------------------------------------------------------


def test_build_stores_every_description_with_its_vector_model_and_hash(built: Any) -> None:
    engine, fake = built

    stored = rows(
        engine,
        "schema_docs",
        "table_name, column_name, embedding_model, content_hash, vector_dims(embedding)",
    )
    assert len(stored) == len(DOCS) == 21
    assert {row.embedding_model for row in stored} == {MODEL}
    assert {row.vector_dims for row in stored} == {384}
    assert all(len(row.content_hash) == 64 for row in stored)
    # What was embedded names the table and the column, so "close" means the price column.
    assert (
        "daily_prices.close: Closing share price in USD, adjusted for stock splits."
        in fake.documents
    )
    assert "daily_prices: Daily share price bars per company from Yahoo Finance." in fake.documents
    assert len(rows(engine, "query_examples", "question, sql")) == 4


def test_a_second_build_embeds_nothing(built: Any) -> None:
    engine, fake = built
    before = len(fake.documents)

    counts = build(engine, fake.embed_documents, MODEL, DOCS, EXAMPLES)

    assert counts == {
        "schema_docs": {"embedded": 0, "unchanged": 21, "removed": 0},
        "query_examples": {"embedded": 0, "unchanged": 4, "sql_updated": 0, "removed": 0},
    }
    assert len(fake.documents) == before


def test_only_changed_rows_are_embedded_again_and_dropped_ones_are_removed(built: Any) -> None:
    engine, fake = built
    before = len(fake.documents)
    reworded = SchemaDoc("daily_prices", "volume", "Shares that changed hands on the day.")
    docs = [
        d
        for d in DOCS
        if d.table != "dividends" and (d.table, d.column) != ("daily_prices", "volume")
    ]
    examples = [
        EXAMPLES[0],
        Example(EXAMPLES[1].question, "SELECT close FROM daily_prices LIMIT 1"),  # same question
        Example("A new question about revenue?", "SELECT 1"),
    ]

    counts = build(engine, fake.embed_documents, MODEL, [*docs, reworded], examples)

    assert counts == {
        "schema_docs": {"embedded": 1, "unchanged": 16, "removed": 4},
        "query_examples": {"embedded": 1, "unchanged": 1, "sql_updated": 1, "removed": 2},
    }
    assert fake.documents[before:] == [reworded.text, "A new question about revenue?"]
    stored = dict(tuple(row) for row in rows(engine, "query_examples", "question, sql"))
    assert stored[EXAMPLES[1].question] == "SELECT close FROM daily_prices LIMIT 1"
    assert not [
        r for r in rows(engine, "schema_docs", "table_name, column_name") if r[0] == "dividends"
    ]


def test_a_truncated_file_is_refused_and_changes_nothing(built: Any) -> None:
    engine, fake = built

    with pytest.raises(BuildError, match="18 of 21 rows would be removed from schema_docs"):
        build(engine, fake.embed_documents, MODEL, DOCS[:3], EXAMPLES)
    with pytest.raises(BuildError, match="4 of 4 rows would be removed from query_examples"):
        build(engine, fake.embed_documents, MODEL, DOCS, [])

    assert len(rows(engine, "schema_docs", "table_name, column_name")) == 21
    assert len(rows(engine, "query_examples", "question, sql")) == 4
    counts = build(engine, fake.embed_documents, MODEL, DOCS, [], prune=True)
    assert counts["query_examples"]["removed"] == 4


def test_the_schema_name_must_be_a_plain_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval, "RETRIEVAL_SCHEMA", "public; DROP TABLE x")

    with pytest.raises(ValueError, match="not a plain identifier"):
        retrieval.schema_name()


def test_vectors_from_another_model_are_refused_until_a_rebuild(built: Any) -> None:
    engine, fake = built

    with pytest.raises(BuildError, match="fake:bag-of-words.*--rebuild"):
        build(engine, fake.embed_documents, "other:model", DOCS, EXAMPLES)

    counts = build(engine, fake.embed_documents, "other:model", DOCS, EXAMPLES, rebuild=True)
    assert counts["schema_docs"]["embedded"] == 21 and counts["query_examples"]["embedded"] == 4
    assert {r[0] for r in rows(engine, "schema_docs", "embedding_model, table_name")} == {
        "other:model"
    }


def test_a_model_with_the_wrong_vector_size_changes_nothing(scratch: Any) -> None:
    engine, _ = scratch

    with pytest.raises(BuildError, match="1536 dimensions, not 384"):
        build(engine, lambda texts: [[0.1] * 1536 for _ in texts], MODEL, DOCS, EXAMPLES)

    assert rows(engine, "schema_docs", "table_name, column_name") == []


def test_fake_vectors_put_related_texts_closer(scratch: Any) -> None:
    def similarity(a: str, b: str) -> float:
        return sum(x * y for x, y in zip(embed(a), embed(b), strict=True))

    assert similarity("share price", "Closing share price in USD") > similarity(
        "share price", "Total revenue"
    )
    assert build_embeddings.EMBED_BATCH > 0
