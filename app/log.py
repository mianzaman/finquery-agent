"""JSON logging for the API: one object per line, with the request's fields.

`logger.info("request", extra={"request_id": ..., "latency_ms": ...})` puts the extras in the
object. Values are encoded by `json`, so a quote or a newline in a message cannot break a line.
Callers never pass keys, connection strings or whole prompts.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Attributes every LogRecord has; anything else on a record came from `extra`.
_STANDARD = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Formats a record as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry |= {key: value for key, value in vars(record).items() if key not in _STANDARD}
        if record.exc_info and record.exc_info[0] is not None:
            # The type only: an exception's text can quote SQL, a URL or a request.
            entry["exception"] = record.exc_info[0].__name__
        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Send the root logger's records to stderr as JSON. Calling it again changes nothing."""
    root = logging.getLogger()
    if any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    # Client libraries log request URLs at INFO; only their warnings are wanted here.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "google", "langchain", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
