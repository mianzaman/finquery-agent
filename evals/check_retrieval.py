"""Checks whether retrieval finds the tables each eval question needs.

Run with `uv run python -m evals.check_retrieval [--k 12]` after
`uv run python -m ingestion.build_embeddings`.

For every eval question that has gold SQL, the tables the gold SQL reads are compared with the
tables `search_schema` returns for the question. Table recall is the share of needed tables
that were retrieved; a question is "complete" when all of its tables were. The agent can only
write correct SQL for a question whose tables it was shown, so this number caps its accuracy.

It also checks that no example question repeats an eval question, which would leak answers.
"""

import argparse
import json
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

from evals.run_eval import QUESTIONS_PATH, EvalError, load_questions

EXAMPLES_PATH = Path(__file__).parent / "examples.jsonl"


def gold_tables(sql: str) -> set[str]:
    """The tables and views a query reads, without the names of its own CTEs."""
    tree = sqlglot.parse_one(sql, read="postgres")
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    return {table.name for table in tree.find_all(exp.Table) if table.name not in cte_names}


def normalize_question(question: str) -> str:
    """Lower-case and strip punctuation and spacing, so near-identical questions compare equal."""
    return " ".join("".join(c if c.isalnum() else " " for c in question.lower()).split())


def overlapping_questions(examples: list[str], eval_questions: list[str]) -> list[str]:
    """Example questions that also appear in the eval set."""
    in_eval = {normalize_question(question) for question in eval_questions}
    return [example for example in examples if normalize_question(example) in in_eval]


def read_example_questions(path: Path = EXAMPLES_PATH) -> list[str]:
    """The question of every line in the examples file."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(json.loads(line).get("question", "")) for line in lines]


def recall_report(rows: list[dict[str, object]]) -> dict[str, float | int]:
    """Overall numbers from per-question rows with `needed` and `missing` table sets."""
    needed = sum(len(row["needed"]) for row in rows)  # type: ignore[arg-type]
    missing = sum(len(row["missing"]) for row in rows)  # type: ignore[arg-type]
    complete = sum(not row["missing"] for row in rows)
    return {
        "questions": len(rows),
        "complete": complete,
        "tables_needed": needed,
        "tables_retrieved": needed - missing,
        "table_recall": (needed - missing) / needed if needed else 0.0,
        "question_recall": complete / len(rows) if rows else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Exit code 1 when examples overlap the eval set or on an error."""
    from app.db import retrieval
    from app.db.session import QueryError
    from app.llm import LLMError

    parser = argparse.ArgumentParser(
        description="Table recall of schema retrieval on the eval set."
    )
    parser.add_argument("--k", type=int, default=12, help="descriptions retrieved per question")
    args = parser.parse_args(argv)

    try:
        questions = [q for q in load_questions(QUESTIONS_PATH) if q.gold_sql]
        overlap = overlapping_questions(
            read_example_questions(), [q.question for q in load_questions(QUESTIONS_PATH)]
        )
        rows: list[dict[str, object]] = []
        for question in questions:
            needed = gold_tables(question.gold_sql or "")
            found = {docs.table for docs in retrieval.search_schema(question.question, args.k)}
            rows.append(
                {"id": question.id, "needed": needed, "missing": needed - found, "found": found}
            )
    except (EvalError, LLMError, QueryError, retrieval.RetrievalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not any(row["found"] - {"companies"} for row in rows):  # type: ignore[operator]
        print(
            "error: nothing was retrieved; run `uv run python -m ingestion.build_embeddings`",
            file=sys.stderr,
        )
        return 1
    report = recall_report(rows)
    recall = f"{report['table_recall'] * 100:.1f}%"
    tables = f"{report['tables_retrieved']} of {report['tables_needed']} tables"
    print(f"## Retrieval check (k = {args.k})\n")
    print("| Metric | Value |\n| --- | --- |")
    print(f"| Table recall | {recall} ({tables}) |")
    print(
        f"| Questions with every table retrieved | {report['complete']} of {report['questions']} |"
    )
    print(f"| Example questions that repeat an eval question | {len(overlap)} |")
    missed = [row for row in rows if row["missing"]]
    if missed:
        print("\n| Question | Missing tables | Retrieved |\n| --- | --- | --- |")
        for row in missed:
            missing = ", ".join(sorted(row["missing"]))  # type: ignore[arg-type]
            found = ", ".join(sorted(row["found"]))  # type: ignore[arg-type]
            print(f"| {row['id']} | {missing} | {found} |")
    for question in overlap:
        print(f"\nOVERLAP: {question}")
    return 1 if overlap else 0


if __name__ == "__main__":
    sys.exit(main())
