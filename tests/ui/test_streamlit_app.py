"""Tests for the Streamlit page: the API is faked at `httpx.post`, so nothing is sent."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).parent.parent.parent / "ui" / "streamlit_app.py")
ANSWER = {
    "answer": "Revenue was 76.4 billion USD.",
    "sql": "SELECT revenue FROM income_statements LIMIT 500",
    "columns": ["period_end", "revenue"],
    "rows": [["2025-06-30", 76441000000]],
    "refused": False,
    "refusal_reason": None,
    "attempts": 1,
    "faithfulness_ok": True,
    "latency_ms": 2300,
    "request_id": "abc",
}
REFUSAL = ANSWER | {
    "answer": None,
    "sql": None,
    "columns": [],
    "rows": [],
    "refused": True,
    "refusal_reason": "That is investment advice.",
    "attempts": 0,
    "faithfulness_ok": None,
}


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake `httpx.post`: answers with `reply` (a Response or an exception) and records calls."""
    control: dict[str, Any] = {"reply": httpx.Response(200, json=ANSWER), "calls": []}

    def post(url: str, **kwargs: Any) -> httpx.Response:
        control["calls"].append((url, kwargs.get("json")))
        control["timeout"] = kwargs.get("timeout")
        if isinstance(control["reply"], Exception):
            raise control["reply"]
        return control["reply"]

    monkeypatch.setattr(httpx, "post", post)
    return control


@pytest.fixture
def page(api: dict[str, Any]) -> AppTest:
    app = AppTest.from_file(PAGE).run()
    assert not app.exception
    return app


def ask(page: AppTest, question: str = "What was Microsoft's revenue?") -> AppTest:
    page.chat_input[0].set_value(question).run()
    assert not page.exception
    return page


def test_the_page_opens_with_six_example_questions_and_no_request(
    page: AppTest, api: dict[str, Any]
) -> None:
    assert page.title[0].value == "FinQuery Agent"
    assert len(page.button) == 6
    assert len({button.label for button in page.button}) == 6
    assert api["calls"] == []


def test_an_answer_shows_the_text_the_sql_the_rows_and_a_caption(
    page: AppTest, api: dict[str, Any]
) -> None:
    ask(page)

    assert api["calls"] == [
        ("http://localhost:8000/ask", {"question": "What was Microsoft's revenue?"})
    ]
    assert [m.value for m in page.markdown] == [
        "What was Microsoft's revenue?",
        "Revenue was 76.4 billion USD.",
    ]
    assert [code.value for code in page.code] == ["SELECT revenue FROM income_statements LIMIT 500"]
    assert page.expander[0].label == "SQL"
    assert page.dataframe[0].value.to_dict("records") == [
        {"period_end": "2025-06-30", "revenue": 76441000000}
    ]
    assert page.caption[-1].value == "2.3 s · 1 repair · numbers checked against the rows"


def test_the_ui_waits_as_long_as_ui_timeout_seconds_says(
    api: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings, get_settings

    assert Settings.model_fields["ui_timeout_seconds"].default == 120
    for seconds in ("300", "45"):  # 300 is what compose sets for a local model
        monkeypatch.setenv("UI_TIMEOUT_SECONDS", seconds)
        get_settings.cache_clear()
        try:
            ask(AppTest.from_file(PAGE).run())
        finally:
            get_settings.cache_clear()
        assert api["timeout"] == float(seconds)


def test_an_example_button_asks_its_question(page: AppTest, api: dict[str, Any]) -> None:
    label = page.button[2].label

    page.button[2].click().run()

    assert api["calls"][0][1] == {"question": label}
    assert page.session_state["history"][0]["question"] == label


def test_the_history_keeps_earlier_answers(page: AppTest, api: dict[str, Any]) -> None:
    ask(page, "First question?")
    api["reply"] = httpx.Response(200, json=REFUSAL)
    ask(page, "Should I buy it?")

    assert [entry["question"] for entry in page.session_state["history"]] == [
        "First question?",
        "Should I buy it?",
    ]
    assert len(page.chat_message) == 4
    assert [info.value for info in page.info] == ["That is investment advice."]
    assert len(page.dataframe) == 1  # the first answer's table is still shown


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (
            httpx.Response(429, json={"code": "rate_limited", "message": "Try again in 40 s."}),
            "Try again in 40 s.",
        ),
        (
            httpx.Response(503, json={"code": "model_rate_limited", "message": "Daily limit."}),
            "Daily limit.",
        ),
        (httpx.Response(500, text="<html>oops</html>"), "could not read"),
        (httpx.Response(200, json=["not", "an", "object"]), "could not read"),
        (httpx.ConnectError("refused"), "could not be reached"),
        (httpx.ReadTimeout("slow"), "took too long"),
    ],
)
def test_api_failures_are_shown_as_a_message(
    page: AppTest, api: dict[str, Any], reply: Any, message: str
) -> None:
    api["reply"] = reply

    ask(page)

    assert len(page.error) == 1 and message in page.error[0].value
    assert len(page.dataframe) == 0


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (ANSWER | {"attempts": 0}, "2.3 s · no repairs · numbers checked against the rows"),
        (ANSWER | {"attempts": 2, "faithfulness_ok": None}, "2.3 s · 2 repairs"),
        (
            ANSWER | {"attempts": 0, "faithfulness_ok": False, "latency_ms": 950},
            "0.9 s · no repairs · the written answer was withheld: "
            "its numbers did not match the rows",
        ),
    ],
)
def test_the_caption(page: AppTest, api: dict[str, Any], result: dict, expected: str) -> None:
    api["reply"] = httpx.Response(200, json=result)

    ask(page)

    assert page.caption[-1].value == expected


def test_the_page_starts_the_way_streamlit_runs_it() -> None:
    """`streamlit run` does not put the project root on the import path; the page must cope."""
    import os
    import subprocess
    import sys

    root = Path(PAGE).parent.parent
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (
        "import runpy, sys; sys.path[:] = [p for p in sys.path if p not in ('', sys.argv[1])];"
        "sys.modules.pop('app', None);"
        "import streamlit; runpy.run_path(sys.argv[2])"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(root), PAGE],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert "ModuleNotFoundError" not in done.stderr, done.stderr[-500:]
    assert done.returncode == 0, done.stderr[-500:]


def test_the_page_never_imports_the_agent() -> None:
    source = Path(PAGE).read_text()

    assert "app.agent" not in source and "app.db" not in source and "app.service" not in source
