"""Retrieve-schema node: finds relevant schema docs and examples.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 08.
"""

import time

from app.agent.state import AgentState, trace_event


def retrieve_schema(state: AgentState) -> dict:
    """Retrieve nothing."""
    started = time.perf_counter()
    return {
        "schema_context": "",
        "examples": [],
        "trace": trace_event("retrieve_schema", started, "stub: nothing retrieved"),
    }
