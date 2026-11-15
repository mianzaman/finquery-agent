"""Fallback-answer node: replaces an answer that failed the faithfulness check twice.

No model. The rows are still returned, so the reader gets the figures without the sentences
that misreported them, and `faithfulness_failed` records that this happened.
"""

import time

from app.agent.state import AgentState, trace_event


def fallback_answer(state: AgentState) -> dict:
    """Say only what is certain: how many rows the query returned."""
    started = time.perf_counter()
    count = len(state.get("rows") or [])
    rows = "1 row" if count == 1 else f"{count} rows"
    return {
        "answer": f"The query returned {rows}; the figures are in the table.",
        "faithfulness_failed": True,
        "trace": trace_event("fallback_answer", started, "the answers had unsupported numbers"),
    }
