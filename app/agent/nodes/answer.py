"""Answer node: turns result rows into a text answer.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 10.
"""

import time

from app.agent.state import AgentState, trace_event


def answer(state: AgentState) -> dict:
    """Say how many rows came back, or that the query could not be made to work."""
    started = time.perf_counter()
    error = state.get("validation_error") or state.get("db_error")
    if error:
        repairs = state.get("attempts", 0)
        text = f"I could not answer this: the query still failed after {repairs} repairs ({error})."
        return {"answer": text, "trace": trace_event("answer", started, "stub: gave up")}
    row_count = len(state.get("rows", []))
    text = f"Stub answer: the query returned {row_count} rows."
    return {"answer": text, "trace": trace_event("answer", started, "stub: answered")}
