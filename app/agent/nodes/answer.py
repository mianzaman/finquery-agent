"""Answer node: turns the query's rows into 1 to 3 sentences.

One model call, as plain text, when there are rows to report. Two cases need no model and get a
fixed sentence: the query still failing after the repairs, and a query that returned no rows.
The failure sentence does not quote the error (it can name tables, columns or the role); the
error stays in `validation_error` or `db_error` and in the trace.

The model's answer is not trusted: the graph sends it to `check_answer`, and may come back here
once with the numbers that check could not find in the rows.
"""

import time
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.nodes.generate_sql import prompt_context
from app.agent.prompts import ANSWER_PROMPT, ANSWER_RETRY_NOTE
from app.agent.state import AgentState, trace_event
from app.llm import invoke, message_text, strip_reasoning

MAX_PROMPT_ROWS = 30  # rows shown to the model; the faithfulness check reads all of them
MAX_ANSWER_CHARS = 1000  # 1 to 3 sentences are far shorter; a longer reply is cut
GAVE_UP = "I could not answer this: the query still failed after {repairs}."
NO_ROWS = "No data was found for {subject}: the query returned no rows."


def format_value(value: Any) -> str:
    """A cell as the prompt shows it: no exponent, no padding zeros, NULL for a missing value."""
    if value is None:
        return "NULL"
    if isinstance(value, Decimal):
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def format_rows(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> tuple[str, str]:
    """The result as `a | b` lines under a header, and a note saying how many rows that is."""
    shown = rows[:MAX_PROMPT_ROWS]
    lines = [" | ".join(columns)]
    lines += [" | ".join(format_value(value) for value in row) for row in shown]
    if len(rows) > len(shown):
        note = f"the first {len(shown)} of {len(rows)} rows"
    else:
        note = "1 row" if len(rows) == 1 else f"{len(rows)} rows"
    return "\n".join(lines), note


def _no_rows_sentence(state: AgentState, companies: str) -> str:
    named = bool(state.get("tickers"))
    period = " in the period asked about" if state.get("periods") else ""
    subject = f"{companies}{period}" if named else f"this question{period}"
    return NO_ROWS.format(subject=subject)


def answer(state: AgentState) -> dict:
    """Write the answer, or the fixed sentence for a failed query or an empty result."""
    started = time.perf_counter()
    error = state.get("validation_error") or state.get("db_error")
    if error:
        count = state.get("attempts", 0)
        text = GAVE_UP.format(repairs="1 repair" if count == 1 else f"{count} repairs")
        note = f"gave up: {error}"[:160]
        return {"answer": text, "trace": trace_event("answer", started, note)}
    rows = state.get("rows") or []
    context = prompt_context(state)
    if not rows:
        text = _no_rows_sentence(state, context["companies"])
        return {"answer": text, "trace": trace_event("answer", started, "no rows")}

    table, row_note = format_rows(state.get("columns") or [], rows)
    prompt = ANSWER_PROMPT.format(
        question=context["question"],
        companies=context["companies"],
        periods=context["periods"],
        row_note=row_note,
        table=table,
    )
    rejected = state.get("unsupported_numbers") or []
    if rejected:
        prompt += ANSWER_RETRY_NOTE.format(numbers=", ".join(rejected))
    text = " ".join(strip_reasoning(message_text(invoke([HumanMessage(prompt)]))).split())
    if len(text) > MAX_ANSWER_CHARS:
        text = text[: MAX_ANSWER_CHARS - 1].rstrip() + "…"
    attempts = state.get("answer_attempts", 0) + 1
    note = f"attempt {attempts}" + (
        f", after unsupported: {', '.join(rejected)}" if rejected else ""
    )
    return {
        "answer": text,
        "answer_attempts": attempts,
        "trace": trace_event("answer", started, note[:160]),
    }
