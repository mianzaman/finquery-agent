"""Entities node: resolves companies and periods without an LLM.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 07.
"""

import time

from app.agent.state import AgentState, trace_event


def entities(state: AgentState) -> dict:
    """Resolve nothing."""
    started = time.perf_counter()
    return {
        "tickers": [],
        "periods": [],
        "trace": trace_event("entities", started, "stub: none resolved"),
    }
