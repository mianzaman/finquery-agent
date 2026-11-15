"""Intent node: classifies the kind of question.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 07.
"""

import time

from app.agent.state import AgentState, trace_event


def intent(state: AgentState) -> dict:
    """Call every question a lookup."""
    started = time.perf_counter()
    return {"intent": "lookup", "trace": trace_event("intent", started, "stub: lookup")}
