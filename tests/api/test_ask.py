"""Tests for `POST /ask`: the agent is replaced, so no model or database is used."""

import json
import logging
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import service
from app.llm import LLMConfigError, LLMError, RateLimitedError, StructuredOutputError
from app.main import create_app
from app.service import json_value

D = Decimal
ANSWERED: dict[str, Any] = {
    "question": "q",
    "refused": False,
    "sql": "SELECT period_end, revenue FROM income_statements LIMIT 500",
    "columns": ["period_end", "revenue", "margin"],
    "rows": [(date(2025, 6, 30), D("76441000000.0000"), D("0.4621"))],
    "attempts": 1,
    "answer": "Revenue was 76.4 billion USD.",
    "answer_attempts": 1,
    "unsupported_numbers": [],
    "faithfulness_failed": False,
}
REFUSED: dict[str, Any] = {
    "question": "q",
    "refused": True,
    "refusal_reason": "That is investment advice.",
    "attempts": 0,
}
FAILED: dict[str, Any] = {
    "question": "q",
    "refused": False,
    "sql": "SELECT nope FROM companies",
    "db_error": 'permission denied for table "secrets" (role agent_reader)',
    "attempts": 2,
    "answer": "I could not answer this: the query still failed after 2 repairs.",
}
QUESTION = {"question": "What was Microsoft's revenue last quarter?"}


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stands in for the graph: returns `state`, or raises `error`, and records the questions."""
    control: dict[str, Any] = {"state": ANSWERED, "error": None, "asked": []}

    def run_agent(question: str) -> dict[str, Any]:
        control["asked"].append(question)
        if control["error"] is not None:
            raise control["error"]
        return control["state"]

    monkeypatch.setattr(service, "run_agent", run_agent)
    return control


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def test_an_answer_has_the_text_the_sql_and_the_rows(agent: dict, client: TestClient) -> None:
    response = client.post("/ask", json={"question": f"  {QUESTION['question']}  "})

    assert response.status_code == 200
    body = response.json()
    request_id = body.pop("request_id")
    assert body.pop("latency_ms") >= 0
    assert body == {
        "answer": "Revenue was 76.4 billion USD.",
        "sql": "SELECT period_end, revenue FROM income_statements LIMIT 500",
        "columns": ["period_end", "revenue", "margin"],
        "rows": [["2025-06-30", 76441000000, 0.4621]],
        "refused": False,
        "refusal_reason": None,
        "attempts": 1,
        "faithfulness_ok": True,
    }
    assert len(request_id) == 32 and response.headers["x-request-id"] == request_id
    assert agent["asked"] == [QUESTION["question"]]  # stripped before it reaches the agent


def test_a_refusal_is_a_normal_response_without_sql_or_rows(
    agent: dict, client: TestClient
) -> None:
    agent["state"] = REFUSED | {"sql": "SELECT 1", "rows": [(1,)], "columns": ["x"]}

    body = client.post("/ask", json=QUESTION).json()

    assert (body["refused"], body["refusal_reason"]) == (True, "That is investment advice.")
    assert (body["answer"], body["sql"], body["columns"], body["rows"]) == (None, None, [], [])
    assert body["faithfulness_ok"] is None


def test_a_query_that_never_worked_does_not_show_the_database_error(
    agent: dict, client: TestClient
) -> None:
    agent["state"] = FAILED

    response = client.post("/ask", json=QUESTION)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I could not answer this: the query still failed after 2 repairs."
    assert (body["attempts"], body["rows"], body["faithfulness_ok"]) == (2, [], None)
    assert "secrets" not in response.text and "agent_reader" not in response.text


def test_a_withheld_answer_is_reported(agent: dict, client: TestClient) -> None:
    agent["state"] = ANSWERED | {"faithfulness_failed": True, "unsupported_numbers": ["$9"]}

    assert client.post("/ask", json=QUESTION).json()["faithfulness_ok"] is False


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"question": ""}, "question: String should have at least 3 characters"),
        ({"question": "   \n "}, "question: String should have at least 3 characters"),
        ({"question": "ab"}, "question: String should have at least 3 characters"),
        ({"question": "x" * 501}, "question: String should have at most 500 characters"),
        ({}, "question: Field required"),
        ({"question": 5}, "question: Input should be a valid string"),
    ],
)
def test_a_bad_question_is_a_clean_422(
    agent: dict, client: TestClient, payload: dict, message: str
) -> None:
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert (body["code"], body["message"]) == ("invalid_request", message)
    assert body["request_id"] == response.headers["x-request-id"]
    assert agent["asked"] == []


def test_a_body_that_is_not_json_is_a_clean_422(agent: dict, client: TestClient) -> None:
    response = client.post("/ask", content="not json", headers={"content-type": "application/json"})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_every_error_has_the_same_shape(agent: dict, client: TestClient) -> None:
    unreadable = client.post(
        "/ask", content=b"\xff\xfe", headers={"content-type": "application/json"}
    )
    assert unreadable.status_code in (400, 422)  # which one depends on the FastAPI version
    missing = client.get("/nowhere")
    wrong_method = client.get("/ask")
    too_large = client.post("/ask", json={"question": "x" * 20_000})

    for response, status, code in (
        (unreadable, unreadable.status_code, "invalid_request"),
        (missing, 404, "not_found"),
        (wrong_method, 405, "method_not_allowed"),
        (too_large, 413, "request_too_large"),
    ):
        body = response.json()
        assert (response.status_code, body["code"]) == (status, code)
        assert set(body) == {"code", "message", "request_id", "retry_after"}
        assert body["request_id"] == response.headers["x-request-id"]
    assert agent["asked"] == []


def test_only_so_many_questions_are_answered_at_once(agent: dict) -> None:
    """With one slot, a second question that arrives during the first gets a 503."""
    client = TestClient(create_app(max_concurrent=1))
    during: list[Any] = []

    def slow_agent(question: str) -> dict[str, Any]:
        if question == "first one":
            during.append(client.post("/ask", json={"question": "second one"}))
        return ANSWERED

    service.run_agent = slow_agent  # the `agent` fixture restores it

    first = client.post("/ask", json={"question": "first one"})

    (second,) = during
    assert first.status_code == 200
    assert (second.status_code, second.json()["code"]) == (503, "busy")
    assert second.headers["retry-after"] == "5"
    # The slot is free again afterwards, also after a failure.
    assert client.post("/ask", json={"question": "third one"}).status_code == 200


def test_a_failed_question_gives_its_slot_back(agent: dict) -> None:
    client = TestClient(create_app(max_concurrent=1), raise_server_exceptions=False)
    agent["error"] = RuntimeError("boom")
    assert client.post("/ask", json=QUESTION).status_code == 500

    agent["error"] = None
    assert client.post("/ask", json=QUESTION).status_code == 200


def test_a_500_character_question_is_accepted(agent: dict, client: TestClient) -> None:
    assert client.post("/ask", json={"question": "x" * 500}).status_code == 200


def test_the_rate_limit_answers_429_with_the_wait(agent: dict) -> None:
    client = TestClient(create_app(rate_limit=2))

    assert [client.post("/ask", json=QUESTION).status_code for _ in range(3)] == [200, 200, 429]
    response = client.post("/ask", json=QUESTION)
    body = response.json()
    assert (body["code"], body["retry_after"]) == ("rate_limited", 60)
    assert response.headers["retry-after"] == "60"
    assert body["request_id"] == response.headers["x-request-id"]
    assert len(agent["asked"]) == 2  # a limited request never reaches the agent
    # Only /ask is limited.
    assert client.get("/health").status_code == 200 and client.get("/schema").status_code == 200


def test_each_app_has_its_own_limit_from_the_settings(agent: dict) -> None:
    client = TestClient(create_app())  # API_RATE_LIMIT_PER_MINUTE defaults to 20

    codes = [client.post("/ask", json=QUESTION).status_code for _ in range(21)]

    assert codes == [200] * 20 + [429]


@pytest.mark.parametrize(
    ("error", "status", "code", "retry_after", "message"),
    [
        (
            RateLimitedError("The model's daily request limit is reached.", retry_after=12.4),
            503,
            "model_rate_limited",
            12,
            "The model's daily request limit is reached.",
        ),
        (
            RateLimitedError("The model's daily request limit is reached."),
            503,
            "model_rate_limited",
            60,
            "The model's daily request limit is reached.",
        ),
        (
            LLMError("the model request failed (HTTP 503)"),
            502,
            "model_error",
            None,
            "the model request failed (HTTP 503)",
        ),
        (
            StructuredOutputError("the model's reply did not fit: {'sql': 'SELECT hunter2'}"),
            502,
            "model_error",
            None,
            "The model's reply could not be used. Try again.",
        ),
        (
            LLMConfigError("OPENROUTER_API_KEY is not set; LLM_PROVIDER=openrouter needs it"),
            500,
            "model_not_configured",
            None,
            "The model is not set up on the server.",
        ),
        (
            RuntimeError("connection to postgresql://finquery:hunter2@db:5432 failed"),
            500,
            "internal_error",
            None,
            "Something went wrong on the server.",
        ),
    ],
)
def test_failures_become_clean_json(
    agent: dict,
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status: int,
    code: str,
    retry_after: int | None,
    message: str,
) -> None:
    agent["error"] = error

    with caplog.at_level(logging.INFO):
        response = client.post("/ask", json=QUESTION)

    assert response.status_code == status
    body = response.json()
    assert body == {
        "code": code,
        "message": message,
        "request_id": response.headers["x-request-id"],
        "retry_after": retry_after,
    }
    assert response.headers.get("retry-after") == (str(retry_after) if retry_after else None)
    assert "hunter2" not in response.text and "Traceback" not in response.text
    assert "OPENROUTER_API_KEY" not in response.text
    (line,) = [r for r in caplog.records if r.getMessage() == "request"]
    assert (line.status, line.path, line.request_id) == (status, "/ask", body["request_id"])


def test_one_log_line_per_request_carries_the_id_latency_and_attempts(
    agent: dict, client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.post("/ask", json=QUESTION)

    (record,) = [r for r in caplog.records if r.getMessage() == "request"]
    assert record.request_id == response.headers["x-request-id"]
    assert (record.method, record.path, record.status) == ("POST", "/ask", 200)
    assert (record.attempts, record.refused, record.faithfulness_ok, record.rows) == (
        1,
        False,
        True,
        1,
    )
    assert record.latency_ms >= 0
    assert QUESTION["question"] not in json.dumps(record.__dict__, default=str)


def test_cors_allows_the_ui_origin_and_no_other(agent: dict, client: TestClient) -> None:
    allowed = client.post("/ask", json=QUESTION, headers={"origin": "http://localhost:8501"})
    other = client.post("/ask", json=QUESTION, headers={"origin": "http://evil.example"})
    invalid = client.post("/ask", json={}, headers={"origin": "http://localhost:8501"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "access-control-allow-origin" not in other.headers
    assert invalid.status_code == 422  # errors carry the header too
    assert invalid.headers["access-control-allow-origin"] == "http://localhost:8501"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (7, 7),
        ("AAPL", "AAPL"),
        (D("76441000000.0000"), 76441000000),
        (D("0.4621"), 0.4621),
        (D("-1.50"), -1.5),
        (D("NaN"), None),
        (float("nan"), None),
        (float("inf"), None),
        (1.5, 1.5),
        (date(2025, 6, 30), "2025-06-30"),
        ((1, D("2.5")), [1, 2.5]),
        (b"x", "b'x'"),
    ],
)
def test_json_value(value: Any, expected: Any) -> None:
    assert json_value(value) == expected
