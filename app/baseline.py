"""Naive single-prompt text-to-SQL baseline, used for comparison in the evals.

One model call sees the whole schema and the question and returns either a query or a refusal.
There is no retrieval, no examples and no repair: this is the floor the agent has to beat.
The query still goes through `check_sql` and runs as the read-only role, like all model SQL.

Try one question with `uv run python -m app.baseline "What was Apple's revenue in fiscal 2025?"`.
"""

import argparse
import json
import sys
import time
from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.nodes.validate_sql import SqlRejected, check_sql
from app.agent.prompts import BASELINE_PROMPT
from app.db.session import QueryError, run_select
from app.llm import LLMError, RateLimitedError, structured, track_usage

SCHEMA_QUERY = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position
"""


class BaselineOutput(BaseModel):
    """What the model must return: a query, or the reason there is none."""

    sql: str | None = Field(None, description="One PostgreSQL SELECT statement, or null")
    refusal: str | None = Field(
        None, description="Why the question cannot be answered from this database, or null"
    )


class BaselineResult(TypedDict):
    """The outcome of one question. `refusal` is set when the model declined to write SQL."""

    sql: str | None
    refusal: str | None
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    error: str | None
    latency_ms: int
    tokens: int
    requests: int
    cached: int  # replies taken from the local cache: they cost no request, tokens or time
    cost_usd: float | None


@lru_cache
def schema_text() -> str:
    """Every table and view the read-only role can query, one per line, with column types.

    It is read through that role, so the prompt lists exactly what a generated query may use.
    """
    _, rows, _ = run_select(SCHEMA_QUERY)
    tables: dict[str, list[str]] = {}
    for table, column, data_type in rows:
        tables.setdefault(table, []).append(f"{column} {data_type}")
    return "\n".join(f"{table}({', '.join(columns)})" for table, columns in tables.items())


def answer(question: str) -> BaselineResult:
    """Turn a question into SQL with one model call and run it.

    A rate limit is raised as `RateLimitedError`, so a batch can stop; every other failure is
    returned in `error`.
    """
    started = time.perf_counter()
    result = BaselineResult(
        sql=None,
        refusal=None,
        columns=[],
        rows=[],
        truncated=False,
        error=None,
        latency_ms=0,
        tokens=0,
        requests=0,
        cached=0,
        cost_usd=0.0,
    )
    prompt = BASELINE_PROMPT.format(schema=schema_text(), question=question)
    with track_usage() as usage:
        try:
            output = structured(BaselineOutput, [HumanMessage(prompt)])
        except RateLimitedError:
            raise
        except LLMError as exc:
            output = None
            result["error"] = f"model: {exc}"

    if output is not None and output.sql and output.sql.strip():
        result["sql"] = output.sql.strip()
        try:
            columns, rows, truncated = run_select(check_sql(result["sql"]))
            result.update(columns=columns, rows=rows, truncated=truncated)
        except SqlRejected as exc:
            result["error"] = f"rejected: {exc}"
        except QueryError as exc:
            result["error"] = f"database: {exc}"
    elif output is not None:
        result["refusal"] = (output.refusal or "").strip() or "the model returned no SQL"

    result.update(
        latency_ms=round((time.perf_counter() - started) * 1000),
        tokens=usage.total_tokens,
        requests=usage.requests,
        cached=usage.cached,
        cost_usd=usage.cost_usd,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Answer one question from the command line and print the result as JSON."""
    parser = argparse.ArgumentParser(description="Ask the naive baseline one question.")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    try:
        result = answer(args.question)
    except (LLMError, QueryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    shown = dict(result) | {"rows": result["rows"][:20], "row_count": len(result["rows"])}
    print(json.dumps(shown, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
