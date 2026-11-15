"""Intent node: classifies the kind of question.

It makes no model call: the guardrail's call already named the intent. This node only checks
that label and falls back to `lookup` when it is missing or not one of the known kinds.
"""

import time

from app.agent.state import INTENTS, AgentState, trace_event

DEFAULT_INTENT = "lookup"


def intent(state: AgentState) -> dict:
    """Keep the guardrail's intent when it is a known kind; otherwise use `lookup`."""
    started = time.perf_counter()
    given = state.get("intent")
    label = str(given).strip().lower().replace("-", "_").replace(" ", "_") if given else ""
    if label in INTENTS:
        return {"intent": label, "trace": trace_event("intent", started, label)}
    why = f"unknown intent {given!r}" if given else "no intent from the guardrail"
    note = f"{why}; using {DEFAULT_INTENT}"
    return {"intent": DEFAULT_INTENT, "trace": trace_event("intent", started, note)}
