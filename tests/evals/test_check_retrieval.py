"""Tests for the retrieval check and the guard against examples that repeat eval questions."""

from pathlib import Path

import pytest

from evals.check_retrieval import (
    EXAMPLES_PATH,
    gold_tables,
    normalize_question,
    overlapping_questions,
    read_example_questions,
    recall_report,
)
from evals.run_eval import QUESTIONS_PATH, load_questions


@pytest.mark.parametrize(
    ("sql", "tables"),
    [
        ("SELECT revenue FROM income_statements WHERE ticker = 'AAPL'", {"income_statements"}),
        (
            "SELECT i.ticker FROM income_statements i JOIN companies c ON c.ticker = i.ticker",
            {"income_statements", "companies"},
        ),
        (
            "WITH latest AS (SELECT ticker, max(trade_date) d FROM daily_prices GROUP BY 1) "
            "SELECT p.close FROM daily_prices p JOIN latest l ON l.ticker = p.ticker",
            {"daily_prices"},  # the CTE's name is not a table
        ),
        (
            "SELECT (SELECT close FROM daily_prices LIMIT 1) / "
            "(SELECT SUM(eps_diluted_adjusted) FROM income_statements_split_adjusted)",
            {"daily_prices", "income_statements_split_adjusted"},
        ),
        (
            "SELECT amount FROM public.dividends UNION SELECT ratio FROM stock_splits",
            {"dividends", "stock_splits"},
        ),
        ("SELECT 1", set()),
    ],
)
def test_gold_tables(sql: str, tables: set[str]) -> None:
    assert gold_tables(sql) == tables


def test_recall_counts_tables_and_complete_questions() -> None:
    rows = [
        {"id": "q1", "needed": {"a"}, "missing": set()},
        {"id": "q2", "needed": {"a", "b"}, "missing": {"b"}},
        {"id": "q3", "needed": {"a", "b", "c"}, "missing": set()},
    ]

    report = recall_report(rows)

    assert (report["tables_needed"], report["tables_retrieved"]) == (6, 5)
    assert report["table_recall"] == pytest.approx(5 / 6)
    assert (report["complete"], report["questions"]) == (2, 3)
    assert recall_report([])["table_recall"] == 0.0


@pytest.mark.parametrize(
    ("example", "eval_question", "same"),
    [
        ("What was Apple's revenue?", "What was Apple's revenue?", True),
        ("what was apples revenue", "What was Apple's revenue?", False),  # different words
        ("What was  Apple's revenue ?", "what was apple's revenue", True),  # case and spacing
        ("What was Alphabet's revenue?", "What was Apple's revenue?", False),
    ],
)
def test_normalize_question(example: str, eval_question: str, same: bool) -> None:
    assert (normalize_question(example) == normalize_question(eval_question)) is same


def test_overlapping_questions_finds_repeats() -> None:
    examples = ["What was Apple's revenue?", "What was Tesla's price?"]
    eval_questions = ["what was apple's revenue", "Something else?"]

    assert overlapping_questions(examples, eval_questions) == ["What was Apple's revenue?"]
    assert overlapping_questions(examples, []) == []


def test_read_example_questions(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"id": "e01", "question": "One?", "sql": "SELECT 1"}\n\n'
        '{"question": "Two?", "sql": "SELECT 2"}\n'
    )

    assert read_example_questions(path) == ["One?", "Two?"]
    assert read_example_questions(tmp_path / "missing.jsonl") == []


def test_no_example_question_repeats_an_eval_question() -> None:
    """The guard: an example the agent is shown must never be a question it is tested on."""
    if not QUESTIONS_PATH.read_text().strip():
        pytest.skip("evals/questions.jsonl is empty")
    eval_questions = [q.question for q in load_questions(QUESTIONS_PATH)]

    overlap = overlapping_questions(read_example_questions(EXAMPLES_PATH), eval_questions)

    assert overlap == [], f"examples that repeat eval questions: {overlap}"
