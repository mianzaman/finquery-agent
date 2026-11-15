"""Tests for reading the owner's files. Building is tested in tests/db/test_retrieval.py."""

from pathlib import Path

import pytest

from ingestion.build_embeddings import (
    EXAMPLES_PATH,
    SCHEMA_DOCS_PATH,
    BuildError,
    Example,
    SchemaDoc,
    content_hash,
    coverage,
    read_examples,
    read_schema_docs,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_schema_docs_give_a_row_per_table_and_per_column() -> None:
    docs = read_schema_docs(FIXTURES / "schema_docs_sample.yaml")

    assert len(docs) == 4 + 17  # four table descriptions, then their columns
    assert docs[0] == SchemaDoc(
        "companies", None, "One row per company with its ticker, name and sector."
    )
    assert docs[1] == SchemaDoc(
        "companies", "ticker", "Stock ticker symbol, the key used by every other table."
    )
    assert docs[0].text == "companies: One row per company with its ticker, name and sector."
    assert docs[1].text.startswith("companies.ticker: Stock ticker symbol")
    # A folded (>-) description becomes one line.
    income = next(d for d in docs if d.table == "income_statements" and d.column is None)
    assert "\n" not in income.description and income.description.endswith(
        "for each fiscal quarter."
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# only a comment\n", "has no descriptions yet"),
        ("tables: {}\n", "has no descriptions yet"),
        ("other: 1\n", "has no descriptions yet"),
        ("tables:\n  companies: just text\n", "companies must have `description` and `columns`"),
        ("tables:\n  companies:\n    columns:\n      ticker: T\n", "companies has no description"),
        ("tables:\n  companies:\n    description: D\n    columns: [a, b]\n", "columns must map"),
        (
            "tables:\n  companies:\n    description: D\n    columns:\n      ticker:\n",
            "companies.ticker has no description",
        ),
    ],
)
def test_schema_docs_explain_a_bad_file(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "schema_docs.yaml"
    path.write_text(content)

    with pytest.raises(BuildError, match=message):
        read_schema_docs(path)


def test_a_table_may_have_a_description_and_no_columns(tmp_path: Path) -> None:
    path = tmp_path / "schema_docs.yaml"
    path.write_text("tables:\n  companies:\n    description: One row per company.\n")

    assert read_schema_docs(path) == [SchemaDoc("companies", None, "One row per company.")]


def test_examples_read_question_and_sql_and_ignore_the_id() -> None:
    examples = read_examples(FIXTURES / "examples_sample.jsonl")

    assert len(examples) == 4
    assert examples[0] == Example(
        "What was Alphabet's total revenue in fiscal 2024?",
        "SELECT SUM(revenue) FROM income_statements WHERE ticker = 'GOOGL' AND fiscal_year = 2024",
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json\n", "line 1 is not valid JSON"),
        ('{"id": "e01", "question": "Q?"}\n', "line 1 needs a question and its sql"),
        ('{"question": "", "sql": "SELECT 1"}\n', "line 1 needs a question and its sql"),
        (
            '{"question": "Q?", "sql": "SELECT 1"}\n{"question": "Q?", "sql": "SELECT 2"}\n',
            "repeats the question: Q",
        ),
    ],
)
def test_examples_explain_a_bad_file(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(content)

    with pytest.raises(BuildError, match=message):
        read_examples(path)


def test_an_empty_examples_file_is_allowed_and_a_missing_one_is_not(tmp_path: Path) -> None:
    empty = tmp_path / "examples.jsonl"
    empty.write_text("\n")

    assert read_examples(empty) == []
    with pytest.raises(BuildError, match="does not exist"):
        read_examples(tmp_path / "nope.jsonl")
    with pytest.raises(BuildError, match="does not exist"):
        read_schema_docs(tmp_path / "nope.yaml")


def test_content_hash_changes_with_the_text() -> None:
    assert content_hash("a: b") == content_hash("a: b")
    assert content_hash("a: b") != content_hash("a: c")
    assert len(content_hash("")) == 64


def test_coverage_reports_what_is_undescribed_and_what_does_not_exist() -> None:
    schema = {
        "companies": (("ticker", "text"), ("name", "text"), ("cik", "bigint")),
        "dividends": (("ticker", "text"), ("amount", "numeric")),
        "a_view": (("ticker", "text"), ("extra", "numeric")),
    }
    docs = [
        SchemaDoc("companies", None, "d"),
        SchemaDoc("companies", "ticker", "d"),
        SchemaDoc("companies", "name", "d"),
        SchemaDoc("companies", "ceo", "d"),  # no such column
        SchemaDoc("a_view", None, "d"),
        SchemaDoc("a_view", "extra", "d"),  # its ticker is described on companies
        SchemaDoc("old_table", None, "d"),  # no such table
    ]

    assert coverage(docs, schema) == {
        "undescribed": ["dividends", "companies.cik", "dividends.amount"],
        "unknown": ["old_table", "companies.ceo"],
    }


def test_the_owners_files_can_be_read() -> None:
    """Whatever state the owner's files are in, a mistake in them is caught here, not at build."""
    for path, reader in ((SCHEMA_DOCS_PATH, read_schema_docs), (EXAMPLES_PATH, read_examples)):
        try:
            reader(path)
        except BuildError as exc:
            if "no descriptions yet" not in str(exc):
                raise
