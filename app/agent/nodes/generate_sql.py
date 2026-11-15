"""Generate-SQL node: writes a query for the question.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 09.
"""

import time

from app.agent.state import AgentState, trace_event


def generate_sql(state: AgentState) -> dict:
    """Write the same harmless query for every question."""
    started = time.perf_counter()
    return {"sql": "SELECT 1", "trace": trace_event("generate_sql", started, "stub: SELECT 1")}
