"""Tests for the JSON log formatter."""

import json
import logging

from app.log import JsonFormatter, configure_logging


def record(message: str, *args: object, **extra: object) -> logging.LogRecord:
    made = logging.LogRecord("app.main", logging.INFO, __file__, 1, message, args, None)
    for key, value in extra.items():
        setattr(made, key, value)
    return made


def test_a_record_is_one_json_object_with_its_extras() -> None:
    line = JsonFormatter().format(
        record("request", request_id="abc", latency_ms=12, attempts=1, faithfulness_ok=None)
    )

    entry = json.loads(line)
    assert "\n" not in line
    assert entry.pop("time").endswith("+00:00")
    assert entry == {
        "level": "INFO",
        "logger": "app.main",
        "message": "request",
        "request_id": "abc",
        "latency_ms": 12,
        "attempts": 1,
        "faithfulness_ok": None,
    }


def test_quotes_and_newlines_cannot_break_the_line() -> None:
    line = JsonFormatter().format(record('he said "hi"\n{"level": "FAKE"} %s', "x"))

    assert "\n" not in line
    assert json.loads(line)["message"] == 'he said "hi"\n{"level": "FAKE"} x'


def test_an_exception_is_logged_by_type_only() -> None:
    try:
        raise RuntimeError("postgresql://finquery:hunter2@db:5432")
    except RuntimeError:
        import sys

        made = record("unhandled error", request_id="abc")
        made.exc_info = sys.exc_info()

    line = JsonFormatter().format(made)

    assert json.loads(line)["exception"] == "RuntimeError"
    assert "hunter2" not in line and "Traceback" not in line


def test_configure_logging_adds_its_handler_once() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging()
        configure_logging()
        added = [h for h in root.handlers if isinstance(h.formatter, JsonFormatter)]
        assert len(added) == 1
    finally:
        root.handlers[:] = before
