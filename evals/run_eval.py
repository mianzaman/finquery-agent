"""Runs the evaluation set against the baseline or the agent.

Run with `uv run python -m evals.run_eval --system baseline [--ids q001,q002] [--limit N]
[--resume] [--max-requests 40] [--questions PATH] [--provider ollama --model NAME]`, or with
`--compare [--allow-mixed]` to print the latest saved baseline and agent runs side by side (no
model call).

Which model
-----------
`.env` names the model. `--provider` and `--model` replace it for one run (and drop `.env`'s
fallback models). Every results file records the provider and the model; `--resume` refuses to
continue a run made with another model, and `--compare` refuses runs made with different models
unless `--allow-mixed` is passed.

How a question is scored
------------------------
- The gold SQL and the system's SQL are both run as the read-only role and their result sets
  compared (`compare_results`): row order and column names are ignored, numbers match within
  0.5%, NULL matches only NULL, and the system may return extra columns as long as every gold
  column is found among them. The number of rows must be the same.
- An `out_of_scope` question passes only if the system refuses. An in-scope question that is
  refused fails.
- Faithfulness is the agent's own check of its text answer against its rows (every number in
  the answer is in the rows). It is counted over the answers that were checked, and says
  nothing about whether the SQL was right. The baseline writes no text answer, so it has none.

Budget
------
Free models allow few requests a day, so the run first prints how many it expects to make and
stops if that is above `--max-requests` (not for Ollama, which has no quota). A rate limit ends
the run; what was finished is saved and `--resume` continues from it. Results files are never
changed: a resumed run writes a new file that carries the earlier answers over.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args

from app.config import LOG_FORMAT, Settings

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

CATEGORIES = ("lookup", "trend", "comparison", "ranking", "multi_table", "out_of_scope")
OUT_OF_SCOPE = "out_of_scope"
RELATIVE_TOLERANCE = 0.005
ABSOLUTE_TOLERANCE = 1e-9  # so that 0 matches 0.0000001 left by rounding
MAX_COLUMN_MAPPINGS = 5000
SAMPLE_ROWS = 10  # rows of each result kept in the results file
# Model calls a system makes for one question (a JSON retry can add one more).
REQUESTS_PER_QUESTION = {"baseline": 1, "agent": 3}
PROVIDERS = get_args(Settings.model_fields["llm_provider"].annotation)

logger = logging.getLogger(__name__)

Rows = Sequence[Sequence[Any]]
System = Callable[[str], dict[str, Any]]


class EvalError(Exception):
    """The evaluation cannot run as asked. The message says what to change."""


@dataclass(frozen=True)
class Question:
    """One line of the questions file."""

    id: str
    question: str
    gold_sql: str | None
    category: str


# --- Questions --------------------------------------------------------------------------------


def load_questions(path: Path) -> list[Question]:
    """Read and check the questions file. It is never written here: the owner maintains it."""
    if not path.exists():
        raise EvalError(f"{path} does not exist")
    questions: list[Question] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{where} is not valid JSON") from exc
        missing = {"id", "question", "gold_sql", "category"} - set(raw)
        if missing:
            raise EvalError(f"{where} has no {', '.join(sorted(missing))}")
        if raw["category"] not in CATEGORIES:
            raise EvalError(f"{where}: unknown category {raw['category']!r}")
        if (raw["category"] == OUT_OF_SCOPE) != (raw["gold_sql"] is None):
            raise EvalError(f"{where}: gold_sql must be null for out_of_scope questions only")
        questions.append(Question(raw["id"], raw["question"], raw["gold_sql"], raw["category"]))
    duplicates = [name for name, count in Counter(q.id for q in questions).items() if count > 1]
    if duplicates:
        raise EvalError(f"{path.name} repeats the ids {', '.join(sorted(duplicates))}")
    if not questions:
        raise EvalError(
            f"{path} has no questions. The owner writes them, one JSON object per line: "
            '{"id": "q001", "question": "...", "gold_sql": "SELECT ...", "category": "lookup"}'
        )
    return questions


def select_questions(
    questions: list[Question], ids: list[str] | None, limit: int | None, done: set[str]
) -> list[Question]:
    """Apply `--ids`, leave out what a resumed run already answered, then apply `--limit`."""
    if ids:
        unknown = set(ids) - {q.id for q in questions}
        if unknown:
            raise EvalError(f"no such question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in ids]
    questions = [q for q in questions if q.id not in done]
    return questions[:limit] if limit is not None else questions


# --- Comparing result sets --------------------------------------------------------------------


def normalize(value: Any) -> Any:
    """Bring a database value to a form that compares across drivers and SQL styles."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int | float | Decimal):
        return float(value)
    if isinstance(value, datetime) and value.time() == time(0):
        # date_trunc() returns midnight timestamps; they mean the same day as a DATE.
        return value.date().isoformat()
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return tuple(normalize(item) for item in value)
    return str(value)


def values_match(gold: Any, system: Any) -> bool:
    """Equal after normalizing; two numbers are equal within 0.5%. NULL equals only NULL."""
    if isinstance(gold, float) and isinstance(system, float):
        if math.isnan(gold) or math.isnan(system):
            return math.isnan(gold) and math.isnan(system)
        return math.isclose(gold, system, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)
    return type(gold) is type(system) and gold == system


def _sort_key(value: Any) -> tuple[int, Any]:
    # Values of different types never compare, so the type decides first.
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, repr(value))


def _column_matches(gold: list[Any], system: list[Any]) -> bool:
    """True when two columns hold the same values, in any order."""
    pairs = zip(sorted(gold, key=_sort_key), sorted(system, key=_sort_key), strict=True)
    return all(values_match(a, b) for a, b in pairs)


def _row_matches(gold: Sequence[Any], system: Sequence[Any]) -> bool:
    return all(values_match(a, b) for a, b in zip(gold, system, strict=True))


def _rows_match(gold: list[tuple[Any, ...]], system: list[tuple[Any, ...]]) -> bool:
    """True when the two lists hold the same rows, in any order."""

    def row_key(row: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
        return tuple(_sort_key(value) for value in row)

    ordered_gold, ordered_system = sorted(gold, key=row_key), sorted(system, key=row_key)
    if all(_row_matches(a, b) for a, b in zip(ordered_gold, ordered_system, strict=True)):
        return True
    # Sorting can pair rows wrongly when two numbers sit within the tolerance of each other,
    # so pair each gold row with any unused system row that matches.
    unused = list(ordered_system)
    for row in ordered_gold:
        found = next((i for i, other in enumerate(unused) if _row_matches(row, other)), None)
        if found is None:
            return False
        unused.pop(found)
    return True


def compare_results(gold: Rows, system: Rows) -> tuple[bool, str]:
    """Decide whether the system's result set answers the question as the gold one does.

    Returns `(matches, reason)`; the reason explains a mismatch and is empty for a match.
    """
    if len(gold) != len(system):
        return False, f"row count differs: gold {len(gold)}, system {len(system)}"
    if not gold:
        return True, ""
    gold_rows = [tuple(normalize(value) for value in row) for row in gold]
    system_rows = [tuple(normalize(value) for value in row) for row in system]
    gold_width, system_width = len(gold_rows[0]), len(system_rows[0])
    if system_width < gold_width:
        return False, f"too few columns: gold {gold_width}, system {system_width}"

    gold_columns = [list(column) for column in zip(*gold_rows, strict=True)]
    system_columns = [list(column) for column in zip(*system_rows, strict=True)]
    # For each gold column, the system columns that hold the same values.
    candidates = [
        [s for s, column in enumerate(system_columns) if _column_matches(gold_column, column)]
        for gold_column in gold_columns
    ]
    if not all(candidates):
        missing = next(i for i, found in enumerate(candidates) if not found)
        return False, f"no system column holds the values of gold column {missing + 1}"

    tried = 0
    for mapping in _mappings(candidates):
        tried += 1
        projected = [tuple(row[s] for s in mapping) for row in system_rows]
        if _rows_match(gold_rows, projected):
            return True, ""
        if tried >= MAX_COLUMN_MAPPINGS:
            break
    return False, "the columns match one by one, but not row by row"


def _mappings(candidates: list[list[int]], used: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    """Every way to give each gold column a different system column from its candidates."""
    if len(used) == len(candidates):
        yield used
        return
    for column in candidates[len(used)]:
        if column not in used:
            yield from _mappings(candidates, (*used, column))


# --- Scoring ----------------------------------------------------------------------------------


def _sample(rows: Rows) -> list[list[Any]]:
    return [[_json_value(value) for value in row] for row in rows[:SAMPLE_ROWS]]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)  # Decimal, date: exact text


def run_gold(sql: str) -> tuple[Rows, str | None]:
    """Run a gold query through the same checks and role as system SQL."""
    from app.agent.nodes.validate_sql import SqlRejected, check_sql
    from app.db.session import QueryError, run_select

    try:
        _, rows, truncated = run_select(check_sql(sql))
    except SqlRejected as exc:
        return [], f"rejected: {exc}"
    except QueryError as exc:
        return [], f"database: {exc}"
    return rows, "more rows than the comparison limit" if truncated else None


def score(
    question: Question, result: dict[str, Any], gold_rows: Rows, gold_error: str | None
) -> dict[str, Any]:
    """Judge one system result and return the entry stored in the results file."""
    refused = result.get("sql") is None and result.get("error") is None
    in_scope = question.category != OUT_OF_SCOPE
    ran = result.get("sql") is not None and result.get("error") is None
    if not in_scope:
        passed, reason = refused, "" if refused else "should have refused"
    elif gold_error:
        passed, reason = False, f"gold SQL failed ({gold_error})"
    elif refused:
        passed, reason = False, "refused an answerable question"
    elif not ran:
        passed, reason = False, str(result.get("error"))
    elif result.get("truncated"):
        passed, reason = False, "more rows than the comparison limit"
    else:
        passed, reason = compare_results(gold_rows, result.get("rows", []))
    return {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "passed": passed,
        "reason": reason,
        "refused": refused,
        "refusal": result.get("refusal"),
        "valid_sql": ran if in_scope else None,
        "gold_error": gold_error,
        "system_sql": result.get("sql"),
        "error": result.get("error"),
        "gold_row_count": len(gold_rows),
        "system_row_count": len(result.get("rows", [])),
        "gold_sample": _sample(gold_rows),
        "system_sample": _sample(result.get("rows", [])),
        "answer": result.get("answer"),
        "faithfulness": result.get("faithfulness"),
        "attempts": result.get("attempts", 0),
        "answer_attempts": result.get("answer_attempts", 0),
        "latency_ms": result.get("latency_ms", 0),
        "tokens": result.get("tokens", 0),
        "requests": result.get("requests", 0),
        "cached": result.get("cached", 0),
        "cost_usd": result.get("cost_usd"),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _rate(entries: list[dict[str, Any]], key: str = "passed") -> float | None:
    return sum(bool(e[key]) for e in entries) / len(entries) if entries else None


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's metrics. Rates are None when no question of that kind was run."""
    in_scope = [e for e in entries if e["category"] != OUT_OF_SCOPE]
    out_of_scope = [e for e in entries if e["category"] == OUT_OF_SCOPE]
    latencies = [float(e["latency_ms"]) for e in entries]
    costs = [e["cost_usd"] for e in entries]
    # Entries carried over from a file written before these keys existed do not have them.
    checked = [e for e in entries if e.get("faithfulness") is not None]
    repaired = [e for e in in_scope if e.get("attempts", 0) > 0]
    return {
        "questions": len(entries),
        "passed": sum(bool(e["passed"]) for e in entries),
        "overall_accuracy": _rate(entries),
        "execution_accuracy": _rate(in_scope),
        "valid_sql_rate": _rate(in_scope, "valid_sql"),
        "refusal_accuracy": _rate(out_of_scope),
        "answers_checked": len(checked),
        "faithfulness_rate": _rate(checked, "faithfulness"),
        "repairs": sum(e.get("attempts", 0) for e in in_scope),
        "questions_repaired": len(repaired),
        "repair_success": _rate(repaired, "valid_sql"),
        "by_category": {
            category: {
                "questions": len(group),
                "passed": sum(bool(e["passed"]) for e in group),
                "accuracy": _rate(group),
            }
            for category in CATEGORIES
            if (group := [e for e in entries if e["category"] == category])
        },
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "average_tokens": sum(e["tokens"] for e in entries) / len(entries) if entries else None,
        "requests": sum(e["requests"] for e in entries),
        # A cached reply shows 0 tokens and almost no latency, which flatters those two metrics.
        "cached_replies": sum(e.get("cached", 0) for e in entries),
        "cost_usd": None if any(cost is None for cost in costs) else sum(costs),
        "gold_errors": [e["id"] for e in entries if e["gold_error"]],
    }


# --- Report -----------------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:,.0f}{unit}"


def metric_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """The summary as (metric, value) pairs. A summary saved by an older version still works."""
    cost = summary["cost_usd"]
    checked = summary.get("answers_checked", 0)
    repaired = summary.get("questions_repaired", 0)
    faithfulness = f"{_percent(summary.get('faithfulness_rate'))} ({checked} answers checked)"
    repairs = f"{summary.get('repairs', 0)} on {repaired} questions"
    if repaired:
        repairs += f" ({_percent(summary.get('repair_success'))} then ran)"
    return [
        ("Questions", f"{summary['questions']} ({summary['passed']} passed)"),
        ("Execution accuracy (answerable)", _percent(summary["execution_accuracy"])),
        ("Refusal accuracy (out of scope)", _percent(summary["refusal_accuracy"])),
        ("Overall accuracy", _percent(summary["overall_accuracy"])),
        ("Valid SQL rate", _percent(summary["valid_sql_rate"])),
        ("Faithfulness (answer numbers in rows)", faithfulness if checked else "n/a"),
        ("Repairs", repairs),
        (
            "Latency p50 / p95",
            f"{_number(summary['latency_ms_p50'], ' ms')} / "
            f"{_number(summary['latency_ms_p95'], ' ms')}",
        ),
        ("Average tokens per question", _number(summary["average_tokens"])),
        ("Model requests", f"{summary['requests']} (+{summary['cached_replies']} from cache)"),
        ("Estimated cost", "unknown" if cost is None else f"${cost:.4f}"),
    ]


def markdown_comparison(runs: list[dict[str, Any]]) -> str:
    """Saved runs side by side, one column each, with the file every number comes from."""
    columns = [metric_rows(run["summary"]) for run in runs]
    header = " | ".join(run["system"] for run in runs)
    lines = [
        "## Eval comparison",
        "",
        f"| Metric | {header} |",
        "| --- |" + " --- |" * len(runs),
        f"| Results file | {' | '.join(run.get('file', 'unknown') for run in runs)} |",
        f"| Model | {' | '.join(model_label(run) for run in runs)} |",
    ]
    for index, (metric, _) in enumerate(columns[0]):
        lines.append(f"| {metric} | {' | '.join(column[index][1] for column in columns)} |")
    partial = [run["system"] for run in runs if not run["complete"]]
    if partial:
        lines += ["", f"**Not a full run:** {', '.join(partial)} stopped early."]
    sizes = {run["summary"]["questions"] for run in runs}
    if len(sizes) > 1:
        lines += ["", "**The runs cover different numbers of questions; compare with care.**"]
    return "\n".join(lines)


def markdown_summary(run: dict[str, Any]) -> str:
    """The run as Markdown tables, for the terminal and for pasting into notes."""
    summary = run["summary"]
    lines = [
        f"## Eval: {run['system']} ({run['model'] or 'no model'}, {run['finished_at']})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        *(f"| {metric} | {value} |" for metric, value in metric_rows(summary)),
        "",
        "| Category | Passed | Accuracy |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {category} | {group['passed']}/{group['questions']} | {_percent(group['accuracy'])} |"
        for category, group in summary["by_category"].items()
    ]
    failed = [e for e in run["questions"] if not e["passed"]]
    if failed:
        lines += ["", "| Failed | Category | Why |", "| --- | --- | --- |"]
        lines += [f"| {e['id']} | {e['category']} | {e['reason']} |" for e in failed]
    if not run["complete"]:
        lines += ["", f"**Stopped early:** {run['stopped_reason']}"]
    return "\n".join(lines)


# --- Results files ----------------------------------------------------------------------------


def latest_results(results_dir: Path, system: str) -> Path | None:
    """The newest results file of a system (the names sort by time)."""
    files = sorted(results_dir.glob(f"*_{system}.json"))
    return files[-1] if files else None


def model_label(run: dict[str, Any]) -> str:
    """`provider/model` of a saved run; files from before the provider was saved say unknown."""
    return f"{run.get('provider') or 'unknown'}/{run.get('model') or 'unknown'}"


def compare_latest(
    results_dir: Path,
    systems: Sequence[str] = ("baseline", "agent"),
    allow_mixed: bool = False,
) -> str:
    """The comparison table of each system's newest results file.

    Runs made with different models are not comparable, so they are refused unless
    `allow_mixed`; then the table says so.
    """
    runs: list[dict[str, Any]] = []
    for system in systems:
        path = latest_results(results_dir, system)
        if path is None:
            raise EvalError(f"--compare: no {system} results in {results_dir}")
        runs.append(json.loads(path.read_text(encoding="utf-8")) | {"file": path.name})
    labels = [f"{run['file']} used {model_label(run)}" for run in runs]
    mixed = len({model_label(run) for run in runs}) > 1
    if mixed and not allow_mixed:
        raise EvalError(
            "--compare: the latest runs used different models ("
            + "; ".join(labels)
            + "). Run both systems with the same model, or pass --allow-mixed."
        )
    table = markdown_comparison(runs)
    if mixed:
        table += "\n\n**The runs used different models; the numbers are not comparable.**"
    return table


def results_path(results_dir: Path, system: str, now: datetime) -> Path:
    """A new file name for this minute; a letter is added rather than overwriting a file."""
    stamp = now.strftime("%Y-%m-%d_%H%M")
    path = results_dir / f"{stamp}_{system}.json"
    for suffix in "bcdefghijklmnopqrstuvwxyz":
        if not path.exists():
            break
        path = results_dir / f"{stamp}{suffix}_{system}.json"
    return path


def git_commit() -> str | None:
    """The commit the run was made from, so a number can be traced to the code."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


# --- Running ----------------------------------------------------------------------------------


def agent_answer(question: str) -> dict[str, Any]:
    """Run the agent graph on one question and report it in the shape the baseline uses.

    A rate limit is raised, so the run can stop; any other model failure becomes `error`.
    """
    import time as clock

    from app.agent.graph import run_agent
    from app.llm import LLMConfigError, LLMError, RateLimitedError, track_usage

    started = clock.perf_counter()
    state: dict[str, Any] = {}
    error: str | None = None
    with track_usage() as usage:
        try:
            state = dict(run_agent(question))
        except (RateLimitedError, LLMConfigError):
            # Neither is about this question: the next one would fail the same way.
            raise
        except LLMError as exc:
            error = f"model: {exc}"
        except Exception as exc:
            # One question that breaks the agent (database down, a bug in a node) is scored as
            # a failure with its cause, so the run goes on and its results are still saved.
            error = f"agent failed: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    refused = bool(state.get("refused"))
    # The check ran if it left its list; its verdict is "no fallback was needed".
    checked = "unsupported_numbers" in state
    if error is None and not refused:
        failure = state.get("validation_error") or state.get("db_error")
        error = f"query: {failure}" if failure else None
    return {
        "sql": None if refused else state.get("sql"),
        "refusal": state.get("refusal_reason") if refused else None,
        "columns": state.get("columns", []),
        "rows": [] if refused else state.get("rows", []),
        "truncated": False,
        "error": error,
        "answer": state.get("answer"),
        "faithfulness": not state.get("faithfulness_failed", False) if checked else None,
        "attempts": state.get("attempts", 0),
        "answer_attempts": state.get("answer_attempts", 0),
        "trace": state.get("trace", []),
        "latency_ms": round((clock.perf_counter() - started) * 1000),
        "tokens": usage.total_tokens,
        "requests": usage.requests,
        "cached": usage.cached,
        "cost_usd": usage.cost_usd,
    }


def get_system(name: str) -> System:
    """The function that answers a question for the named system."""
    if name == "baseline":
        from app.baseline import answer

        return answer  # type: ignore[return-value]
    if name == "agent":
        return agent_answer
    raise NotImplementedError(f"the {name} system is not built yet")


def evaluate(
    questions: list[Question],
    system: System,
    gold_runner: Callable[[str], tuple[Rows, str | None]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Answer and score each question. Returns the entries and why the run stopped early."""
    from app.llm import RateLimitedError

    gold_runner = gold_runner or run_gold
    entries: list[dict[str, Any]] = []
    for question in questions:
        try:
            result = system(question.question)
        except RateLimitedError as exc:
            return entries, str(exc)
        gold_rows, gold_error = gold_runner(question.gold_sql) if question.gold_sql else ([], None)
        entry = score(question, result, gold_rows, gold_error)
        entries.append(entry)
        logger.info("%s %s %s", question.id, "pass" if entry["passed"] else "FAIL", entry["reason"])
    return entries, None


def use_model(provider: str | None, model: str | None) -> None:
    """Point this process at another provider or model than `.env` names, for this run only.

    Settings read the environment before `.env`, so the override is set there and the cached
    settings are dropped. It must run before the chat model is first built (`main` calls it
    before anything asks for one). `.env`'s fallback models belong to `.env`'s model, so they
    are dropped too: the results file then names the only model that answered.
    """
    from app.config import get_settings

    if provider is not None:
        os.environ["LLM_PROVIDER"] = provider
    if model is not None:
        os.environ["LLM_MODEL"] = model
    os.environ["LLM_FALLBACK_MODELS"] = ""
    get_settings.cache_clear()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Exit code 0: finished; 1: could not run; 2: stopped early."""
    parser = argparse.ArgumentParser(description="Score a system against the gold SQL.")
    parser.add_argument("--system", choices=("baseline", "agent"))
    parser.add_argument(
        "--compare", action="store_true", help="print the latest baseline and agent runs"
    )
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="with --compare: show runs made with different models side by side",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="use this provider for this run instead of LLM_PROVIDER (needs --model)",
    )
    parser.add_argument(
        "--model", help="use this model for this run instead of LLM_MODEL (drops the fallbacks)"
    )
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--limit", type=int, help="run at most this many questions")
    parser.add_argument("--resume", action="store_true", help="continue the latest run")
    parser.add_argument("--max-requests", type=int, default=40, help="refuse a larger run")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.compare:
        try:
            print(compare_latest(args.results_dir, allow_mixed=args.allow_mixed))
        except EvalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.system is None:
        parser.error("--system is required (or use --compare)")
    if args.provider is not None and not args.model:
        parser.error("--provider needs --model: the model in .env belongs to its own provider")
    if args.provider is not None or args.model is not None:
        use_model(args.provider, args.model)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from app.config import get_settings
    from app.db.session import QueryError
    from app.llm import LLMError

    settings = get_settings()
    current = {"provider": settings.llm_provider, "model": settings.llm_model}
    try:
        questions = load_questions(args.questions)
        carried: list[dict[str, Any]] = []
        previous = latest_results(args.results_dir, args.system) if args.resume else None
        if args.resume and previous is None:
            raise EvalError(f"--resume: no earlier {args.system} results in {args.results_dir}")
        if previous is not None:
            earlier = json.loads(previous.read_text(encoding="utf-8"))
            if model_label(earlier) != model_label(current):
                raise EvalError(
                    f"--resume: {previous.name} used {model_label(earlier)}, this run would use "
                    f"{model_label(current)}. Pass the same --provider and --model, or start a "
                    "new run without --resume."
                )
            carried = earlier["questions"]
            known = {q.id for q in questions}
            carried = [e | {"carried_from": previous.name} for e in carried if e["id"] in known]
        ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
        todo = select_questions(questions, ids, args.limit, {e["id"] for e in carried})
        if not todo:
            raise EvalError("nothing to run: every selected question is already answered")

        estimate = len(todo) * REQUESTS_PER_QUESTION[args.system]
        print(f"Model: {model_label(current)}")
        print(
            f"{len(todo)} questions, about {estimate} model requests "
            f"(up to {estimate * 2} if replies need a retry; cached replies cost none)."
        )
        # The budget protects a daily request quota; a local model has none.
        if current["provider"] != "ollama" and estimate > args.max_requests:
            raise EvalError(
                f"{estimate} requests is above --max-requests {args.max_requests}. "
                "Use --limit or --ids to run a batch, or raise --max-requests."
            )
        system = get_system(args.system)
        started = datetime.now().astimezone()
        entries, stopped_reason = evaluate(todo, system)
    except (EvalError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (LLMError, QueryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    finished = datetime.now().astimezone()
    everything = carried + entries
    run = {
        "system": args.system,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "fallback_models": settings.fallback_models,
        "git_commit": git_commit(),
        "questions_file": str(args.questions),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "resumed_from": previous.name if previous else None,
        "complete": stopped_reason is None,
        "stopped_reason": stopped_reason,
        "summary": summarize(everything),
        "questions": everything,
    }
    if everything:
        path = results_path(args.results_dir, args.system, finished)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        print(markdown_summary(run))
        print(f"\nSaved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    if stopped_reason is not None:
        print(f"\nStopped early: {stopped_reason}", file=sys.stderr)
        print(f"Continue with: uv run python -m evals.run_eval --system {args.system} --resume")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
