"""Tests for the evaluation runner. Systems and gold results are fakes: no LLM, no network."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.llm import RateLimitedError
from evals import run_eval
from evals.run_eval import (
    EvalError,
    Question,
    compare_results,
    evaluate,
    latest_results,
    load_questions,
    main,
    markdown_summary,
    percentile,
    results_path,
    score,
    select_questions,
    summarize,
    values_match,
)

D = Decimal

# --- Comparing result sets --------------------------------------------------------------------

MATCHING = {
    "identical": ([("AAPL", 100)], [("AAPL", 100)]),
    "both empty": ([], []),
    "row order is ignored": ([("A", 1), ("B", 2), ("C", 3)], [("C", 3), ("A", 1), ("B", 2)]),
    "column order is ignored": ([("A", 1, 2.5)], [(2.5, "A", 1)]),
    "column names are not part of it": ([(1,)], [(1,)]),
    "within 0.5 percent": ([(1000.0,)], [(1004.9,)]),
    "decimal against float and int": ([(D("391035000000.0000"),)], [(391035000000,)]),
    "rounded ratio": ([(D("46.2064"),)], [(46.21,)]),
    "zero against rounding dust": ([(0,)], [(1e-12,)]),
    "nulls in the same places": ([("A", None), ("B", 2)], [("B", 2), ("A", None)]),
    "an extra system column": ([("A", 1)], [("A", "Apple Inc.", 1)]),
    "extra columns and another order": ([("A", 1), ("B", 2)], [(2, "x", "B"), (1, "y", "A")]),
    "duplicate rows": ([("A", 1), ("A", 1), ("B", 2)], [("B", 2), ("A", 1), ("A", 1)]),
    "dates": ([(date(2025, 9, 27), 1)], [(date(2025, 9, 27), 1)]),
    "a date against midnight of that day": (
        [(date(2025, 1, 1), D("631.65"))],
        [(datetime(2025, 1, 1, tzinfo=UTC), D("631.6545000000000000"))],
    ),
    "two columns with the same values": ([(1, 1), (2, 2)], [(2, 2), (1, 1)]),
    "swapped twin columns still pair up": ([(1, 2), (2, 1)], [(2, 1), (1, 2)]),
    "near-equal numbers sort differently": (
        [(100.0, "x"), (100.2, "y")],
        [(100.3, "x"), (99.9, "y")],
    ),
    "booleans": ([(True, 1)], [(True, 1)]),
}

DIFFERING = {
    "different value": ([("AAPL", 100)], [("AAPL", 200)], "no system column"),
    "outside 0.5 percent": ([(1000.0,)], [(1005.1,)], "no system column"),
    "missing row": ([("A", 1), ("B", 2)], [("A", 1)], "row count"),
    "extra row": ([("A", 1)], [("A", 1), ("B", 2)], "row count"),
    "empty against rows": ([], [("A", 1)], "row count"),
    "missing column": ([("A", 1)], [("A",)], "too few columns"),
    "null against zero": ([(None,)], [(0,)], "no system column"),
    "null against empty text": ([(None,)], [("",)], "no system column"),
    "number against its text": ([(100,)], [("100",)], "no system column"),
    "true against one": ([(True,)], [(1,)], "no system column"),
    "wrong company": ([("AAPL", 100)], [("MSFT", 100)], "no system column"),
    "a date against a time on that day": (
        [(date(2025, 1, 1),)],
        [(datetime(2025, 1, 1, 9, 30),)],
        "no system column",
    ),
    "values paired with the wrong keys": (
        [("A", 1), ("B", 2)],
        [("A", 2), ("B", 1)],
        "not row by row",
    ),
    "duplicates in different numbers": (
        [("A", 1), ("A", 1), ("B", 2)],
        [("A", 1), ("B", 2), ("B", 2)],
        "no system column",
    ),
    "one column cannot serve two gold columns": ([(1, 1)], [(1, "x")], "not row by row"),
}


@pytest.mark.parametrize(("gold", "system"), MATCHING.values(), ids=MATCHING.keys())
def test_results_that_match(gold: list[tuple], system: list[tuple]) -> None:
    assert compare_results(gold, system) == (True, "")


@pytest.mark.parametrize(("gold", "system", "reason"), DIFFERING.values(), ids=DIFFERING.keys())
def test_results_that_differ(gold: list[tuple], system: list[tuple], reason: str) -> None:
    matches, why = compare_results(gold, system)

    assert not matches
    assert reason in why


def test_extra_system_columns_pass() -> None:
    gold = [("NVDA", D("55.85")), ("MSFT", D("36.15"))]
    system = [
        ("MSFT", "Microsoft", D("101832000000"), D("36.15")),
        ("NVDA", "NVIDIA", D("72880000000"), D("55.85")),
    ]

    assert compare_results(gold, system) == (True, "")
    assert compare_results([(D("30711000000"),)], [(2026, 4, date(2026, 9, 26), 30711000000)])[0]


def test_a_missing_gold_column_fails() -> None:
    gold = [("GS", D("13.48")), ("JPM", D("11.21"))]

    # Too few columns, and enough columns but none with the ratio.
    assert compare_results(gold, [("GS",), ("JPM",)]) == (
        False,
        "too few columns: gold 2, system 1",
    )
    matches, reason = compare_results(gold, [("GS", "Goldman Sachs"), ("JPM", "JPMorgan Chase")])
    assert (matches, reason) == (False, "no system column holds the values of gold column 2")


def test_a_percentage_against_a_fraction_fails() -> None:
    gold = [(1, D("46.88")), (2, D("47.05"))]
    fraction = [(1, D("0.46882542236524537409")), (2, D("0.47050619238876246605"))]
    percentage = [(1, D("46.882542236524537409")), (2, D("47.050619238876246605"))]

    assert compare_results(gold, fraction)[0] is False
    assert compare_results(gold, percentage) == (True, "")
    # Extra columns do not rescue it: the percentage itself has to be there.
    assert compare_results(gold, [(q, m, m * 2) for q, m in fraction])[0] is False


def test_a_gold_column_needs_its_own_system_column() -> None:
    # Both gold columns hold 1, but the system has only one column with a 1.
    assert compare_results([(1, 1)], [(1, 2)])[0] is False
    assert compare_results([(1, 1)], [(1, 1, 2)])[0] is True


def test_many_similar_columns_do_not_blow_up() -> None:
    gold = [tuple([row] * 8) for row in range(50)]

    assert compare_results(gold, list(reversed(gold))) == (True, "")


@pytest.mark.parametrize(
    ("gold", "system", "expected"),
    [
        (100.0, 100.5, True),
        (100.0, 100.51, False),
        (-100.0, -100.4, True),
        (-100.0, 100.0, False),
        (0.0, 0.0, True),
        (0.0, 0.001, False),
        (float("nan"), float("nan"), True),
        (float("nan"), 1.0, False),
        (None, None, True),
        (None, 0.0, False),
        ("a", "a", True),
        ("a", "A", False),
        (True, True, True),
        (True, 1.0, False),
    ],
)
def test_values_match(gold: Any, system: Any, expected: bool) -> None:
    assert values_match(gold, system) is expected


# --- Scoring ----------------------------------------------------------------------------------


def question(category: str = "lookup", id: str = "q001") -> Question:
    gold = None if category == "out_of_scope" else "SELECT 1"
    return Question(id, "A question?", gold, category)


def result(**values: Any) -> dict[str, Any]:
    base = {"sql": None, "refusal": None, "rows": [], "error": None, "truncated": False}
    return base | {"latency_ms": 100, "tokens": 500, "requests": 1, "cost_usd": 0.0} | values


REFUSED = result(refusal="That is investment advice.")
ANSWERED = result(sql="SELECT 1", rows=[(1,)])


@pytest.mark.parametrize(
    ("system_result", "passed", "reason"),
    [
        (REFUSED, True, ""),
        (ANSWERED, False, "should have refused"),
        (result(sql="SELECT nope", error="database: column nope"), False, "should have refused"),
        (result(error="model: the model request failed (HTTP 503)"), False, "should have refused"),
    ],
)
def test_out_of_scope_passes_only_on_a_refusal(
    system_result: dict[str, Any], passed: bool, reason: str
) -> None:
    entry = score(question("out_of_scope"), system_result, [], None)

    assert (entry["passed"], entry["reason"]) == (passed, reason)
    assert entry["valid_sql"] is None  # not counted in the valid SQL rate


@pytest.mark.parametrize(
    ("system_result", "gold_error", "passed", "reason", "valid_sql"),
    [
        (ANSWERED, None, True, "", True),
        (result(sql="SELECT 2", rows=[(2,)]), None, False, "no system column", True),
        (REFUSED, None, False, "refused an answerable question", False),
        (
            result(sql="SELECT x", error="database: column x"),
            None,
            False,
            "database: column x",
            False,
        ),
        (result(sql="DROP x", error="rejected: only SELECT"), None, False, "rejected", False),
        (result(error="model: failed"), None, False, "model: failed", False),
        (result(sql="SELECT 1", rows=[(1,)], truncated=True), None, False, "more rows", True),
        (ANSWERED, "database: relation gone", False, "gold SQL failed", True),
    ],
)
def test_answerable_questions(
    system_result: dict[str, Any],
    gold_error: str | None,
    passed: bool,
    reason: str,
    valid_sql: bool,
) -> None:
    entry = score(question("lookup"), system_result, [(1,)], gold_error)

    assert entry["passed"] is passed
    assert reason in entry["reason"]
    assert entry["valid_sql"] is valid_sql


def test_an_entry_can_be_saved_as_json() -> None:
    rows = [(date(2025, 9, 27), D("1.2300"), None, "AAPL")] * 30
    entry = score(question(), result(sql="SELECT 1", rows=rows), rows, None)

    saved = json.loads(json.dumps(entry))
    assert saved["passed"] is True
    assert saved["gold_row_count"] == saved["system_row_count"] == 30
    assert len(saved["system_sample"]) == 10
    assert saved["system_sample"][0] == ["2025-09-27", "1.2300", None, "AAPL"]


# --- Metrics ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [
        ([], 0.5, None),
        ([7.0], 0.95, 7.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.0),
        ([4.0, 1.0, 3.0, 2.0], 0.95, 4.0),
        ([float(n) for n in range(1, 101)], 0.95, 95.0),
    ],
)
def test_percentile(values: list[float], fraction: float, expected: float | None) -> None:
    assert percentile(values, fraction) == expected


def entries() -> list[dict[str, Any]]:
    wrong = result(sql="SELECT 2", rows=[(2,)], latency_ms=300, tokens=700)
    broken = result(sql="SELECT x", error="database: x", latency_ms=200, cost_usd=0.01)
    return [
        score(question("lookup", "q1"), ANSWERED, [(1,)], None),
        score(question("lookup", "q2"), wrong, [(1,)], None),
        score(question("trend", "q3"), broken, [(1,)], None),
        score(question("out_of_scope", "q4"), REFUSED | {"cached": 1}, [], None),
        score(question("out_of_scope", "q5"), ANSWERED, [], None),
    ]


def test_summary_metrics() -> None:
    summary = summarize(entries())

    assert (summary["questions"], summary["passed"]) == (5, 2)
    assert summary["overall_accuracy"] == pytest.approx(2 / 5)
    assert summary["execution_accuracy"] == pytest.approx(1 / 3)
    assert summary["valid_sql_rate"] == pytest.approx(2 / 3)
    assert summary["refusal_accuracy"] == pytest.approx(1 / 2)
    assert summary["by_category"] == {
        "lookup": {"questions": 2, "passed": 1, "accuracy": 0.5},
        "trend": {"questions": 1, "passed": 0, "accuracy": 0.0},
        "out_of_scope": {"questions": 2, "passed": 1, "accuracy": 0.5},
    }
    assert (summary["latency_ms_p50"], summary["latency_ms_p95"]) == (100.0, 300.0)
    assert summary["average_tokens"] == pytest.approx(540)
    assert summary["requests"] == 5
    assert summary["cached_replies"] == 1
    assert summary["cost_usd"] == pytest.approx(0.01)
    assert summary["gold_errors"] == []


def test_summary_of_nothing_and_of_unknown_cost() -> None:
    empty = summarize([])
    assert (empty["questions"], empty["overall_accuracy"], empty["latency_ms_p50"]) == (
        0,
        None,
        None,
    )

    unknown = summarize(
        [score(question(), result(sql="SELECT 1", rows=[(1,)], cost_usd=None), [(1,)], None)]
    )
    assert unknown["cost_usd"] is None
    assert unknown["refusal_accuracy"] is None


def test_markdown_summary_shows_rates_categories_and_failures() -> None:
    run = {
        "system": "baseline",
        "model": "vendor/model:free",
        "finished_at": "2026-11-15T10:00:00+00:00",
        "complete": False,
        "stopped_reason": "limit reached",
        "summary": summarize(entries()),
        "questions": entries(),
    }

    text = markdown_summary(run)

    assert "| Execution accuracy (answerable) | 33.3% |" in text
    assert "| Refusal accuracy (out of scope) | 50.0% |" in text
    assert "| Valid SQL rate | 66.7% |" in text
    assert "| lookup | 1/2 | 50.0% |" in text
    assert "| q5 | out_of_scope | should have refused |" in text
    assert "| Estimated cost | $0.0100 |" in text
    assert "| Model requests | 5 (+1 from cache) |" in text
    assert "**Stopped early:** limit reached" in text


# --- Questions file ---------------------------------------------------------------------------


def write_questions(path: Path, *lines: dict[str, Any] | str) -> Path:
    path.write_text("\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n")
    return path


Q1 = {"id": "q001", "question": "Revenue?", "gold_sql": "SELECT 1", "category": "lookup"}
Q2 = {"id": "q002", "question": "Trend?", "gold_sql": "SELECT 2", "category": "trend"}
Q3 = {"id": "q003", "question": "Buy?", "gold_sql": None, "category": "out_of_scope"}


def test_load_questions(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path / "q.jsonl", Q1, "", Q3))

    assert questions == [
        Question("q001", "Revenue?", "SELECT 1", "lookup"),
        Question("q003", "Buy?", None, "out_of_scope"),
    ]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "has no questions"),
        (["{not json"], "line 1 is not valid JSON"),
        ([Q1, {"id": "q002", "question": "x"}], "line 2 has no category, gold_sql"),
        ([Q1 | {"category": "other"}], "unknown category 'other'"),
        ([Q1 | {"gold_sql": None}], "gold_sql must be null for out_of_scope questions only"),
        ([Q3 | {"gold_sql": "SELECT 1"}], "gold_sql must be null for out_of_scope questions only"),
        ([Q1, Q2 | {"id": "q001"}], "repeats the ids q001"),
    ],
)
def test_load_questions_explains_a_bad_file(tmp_path: Path, lines: list[Any], message: str) -> None:
    with pytest.raises(EvalError, match=message):
        load_questions(write_questions(tmp_path / "q.jsonl", *lines))


def test_load_questions_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="does not exist"):
        load_questions(tmp_path / "nope.jsonl")


def test_select_questions() -> None:
    questions = [question(id=f"q{n}") for n in range(1, 6)]

    def ids(*args: Any) -> list[str]:
        return [q.id for q in select_questions(questions, *args)]

    assert ids(None, None, set()) == ["q1", "q2", "q3", "q4", "q5"]
    assert ids(["q4", "q2"], None, set()) == ["q2", "q4"]
    assert ids(None, 2, set()) == ["q1", "q2"]
    assert ids(None, 2, {"q1", "q2"}) == ["q3", "q4"]  # resume first, then the limit
    with pytest.raises(EvalError, match="no such question ids: q9"):
        ids(["q1", "q9"], None, set())


# --- Results files ----------------------------------------------------------------------------


def test_results_files_are_named_by_time_and_never_overwritten(tmp_path: Path) -> None:
    now = datetime(2026, 11, 15, 9, 5)

    first = results_path(tmp_path, "baseline", now)
    first.write_text("{}")
    second = results_path(tmp_path, "baseline", now)
    second.write_text("{}")

    assert first.name == "2026-11-15_0905_baseline.json"
    assert second.name == "2026-11-15_0905b_baseline.json"
    assert results_path(tmp_path, "agent", now).name == "2026-11-15_0905_agent.json"
    assert latest_results(tmp_path, "baseline") == second
    assert latest_results(tmp_path, "agent") is None


# --- Running ----------------------------------------------------------------------------------


def gold_runner(sql: str) -> tuple[list[tuple], str | None]:
    return [(int(sql.split()[-1]),)], None


def test_evaluate_scores_each_question_and_stops_at_a_rate_limit() -> None:
    questions = [
        Question("q1", "one?", "SELECT 1", "lookup"),
        Question("q2", "two?", "SELECT 2", "trend"),
        Question("q3", "advice?", None, "out_of_scope"),
        Question("q4", "limit?", "SELECT 4", "lookup"),
        Question("q5", "never asked", "SELECT 5", "lookup"),
    ]
    asked: list[str] = []

    def system(text: str) -> dict[str, Any]:
        asked.append(text)
        if text == "limit?":
            raise RateLimitedError("daily limit reached")
        if text == "advice?":
            return REFUSED
        return ANSWERED

    done, stopped = evaluate(questions, system, gold_runner)

    assert [(e["id"], e["passed"]) for e in done] == [("q1", True), ("q2", False), ("q3", True)]
    assert stopped == "daily limit reached"
    assert asked == ["one?", "two?", "advice?", "limit?"]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """`main` wired to a fake system and fake gold results, writing under tmp_path."""
    state: dict[str, Any] = {"asked": [], "limit_at": None, "results": tmp_path / "results"}

    def system(text: str) -> dict[str, Any]:
        if text == state["limit_at"]:
            raise RateLimitedError("daily limit reached")
        state["asked"].append(text)
        return REFUSED if text == "Buy?" else result(sql="SELECT 1", rows=[(1,)])

    monkeypatch.setattr(run_eval, "get_system", lambda name: system)
    monkeypatch.setattr(run_eval, "run_gold", gold_runner)
    state["questions"] = write_questions(tmp_path / "q.jsonl", Q1, Q2, Q3)
    state["args"] = [
        "--system", "baseline",
        "--questions", str(state["questions"]),
        "--results-dir", str(state["results"]),
    ]  # fmt: skip
    return state


def saved_runs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(state["results"].glob("*.json"))]


def test_main_runs_saves_and_prints(harness: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    assert main(harness["args"]) == 0

    (run,) = saved_runs(harness)
    assert run["complete"] is True and run["system"] == "baseline"
    assert [(e["id"], e["passed"]) for e in run["questions"]] == [
        ("q001", True),
        ("q002", False),
        ("q003", True),
    ]
    assert run["summary"]["passed"] == 2
    output = capsys.readouterr().out
    assert "3 questions, about 3 model requests" in output
    assert "| Questions | 3 (2 passed) |" in output
    assert "_baseline.json" in output


def test_main_stops_at_a_rate_limit_and_resumes_into_a_new_file(
    harness: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    harness["limit_at"] = "Trend?"
    assert main(harness["args"]) == 2
    assert "--resume" in capsys.readouterr().out

    harness["limit_at"] = None
    assert main([*harness["args"], "--resume"]) == 0

    first, second = saved_runs(harness)
    assert (first["complete"], [e["id"] for e in first["questions"]]) == (False, ["q001"])
    assert first["stopped_reason"] == "daily limit reached"
    assert [e["id"] for e in second["questions"]] == ["q001", "q002", "q003"]
    assert second["complete"] is True and second["resumed_from"].endswith("_baseline.json")
    assert "carried_from" in second["questions"][0] and "carried_from" not in second["questions"][1]
    assert harness["asked"] == ["Revenue?", "Trend?", "Buy?"]  # q001 was not asked again

    assert main([*harness["args"], "--resume"]) == 1
    assert "already answered" in capsys.readouterr().err


def test_main_refuses_a_run_above_the_request_budget(
    harness: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    assert main([*harness["args"], "--max-requests", "2"]) == 1

    assert "3 requests is above --max-requests 2" in capsys.readouterr().err
    assert harness["asked"] == [] and saved_runs(harness) == []
    assert main([*harness["args"], "--max-requests", "2", "--limit", "2"]) == 0
    assert main([*harness["args"], "--max-requests", "1", "--ids", "q003"]) == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--resume"], "no earlier baseline results"),
        (["--ids", "q999"], "no such question ids"),
    ],
)
def test_main_explains_what_it_cannot_do(
    harness: dict[str, Any], capsys: pytest.CaptureFixture, extra: list[str], message: str
) -> None:
    assert main([*harness["args"], *extra]) == 1
    assert message in capsys.readouterr().err


def test_main_with_an_empty_questions_file(
    harness: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    harness["questions"].write_text("")

    assert main(harness["args"]) == 1
    assert "has no questions" in capsys.readouterr().err


def test_the_agent_system_is_not_built_yet(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    questions = write_questions(tmp_path / "q.jsonl", Q1)

    assert (
        main(["--system", "agent", "--questions", str(questions), "--results-dir", str(tmp_path)])
        == 1
    )
    assert "agent system is not built yet" in capsys.readouterr().err
    with pytest.raises(NotImplementedError):
        run_eval.get_system("agent")


def test_gold_sql_goes_through_the_same_checks(db: None) -> None:
    assert run_eval.run_gold("SELECT 1 AS one") == ([(1,)], None)
    assert run_eval.run_gold("DELETE FROM companies")[1].startswith("rejected: only SELECT")
    assert run_eval.run_gold("SELECT nope FROM companies")[1].startswith("database: column")
