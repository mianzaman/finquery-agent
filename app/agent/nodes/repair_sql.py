"""Repair-SQL node: fixes a query that failed validation or execution.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 09.
"""

import time

from app.agent.state import AgentState, trace_event


def repair_sql(state: AgentState) -> dict:
    """Count the repair and hand the same query back for another check."""
    started = time.perf_counter()
    attempts = state.get("attempts", 0) + 1
    return {
        "sql": state.get("sql"),
        "validation_error": None,
        "db_error": None,
        "attempts": attempts,
        "trace": trace_event("repair_sql", started, f"stub: repair {attempts}"),
    }
