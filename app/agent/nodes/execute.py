"""Execute node: runs validated SQL on the read-only engine.

Stub for the graph skeleton (Task 06): it returns fixed values and sends nothing to the
database. Real logic comes in Task 09.
"""

import time

from app.agent.state import AgentState, trace_event


def execute(state: AgentState) -> dict:
    """Return one fixed row without running the query."""
    started = time.perf_counter()
    return {
        "columns": ["value"],
        "rows": [(1,)],
        "db_error": None,
        "trace": trace_event("execute", started, "stub: 1 fixed row"),
    }
