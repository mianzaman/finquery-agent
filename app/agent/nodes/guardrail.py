"""Guardrail node: refuses out-of-scope questions.

Stub for the graph skeleton (Task 06): it returns fixed values. Real logic comes in Task 07.
"""

import time

from app.agent.state import AgentState, trace_event


def guardrail(state: AgentState) -> dict:
    """Let every question through."""
    started = time.perf_counter()
    return {
        "refused": False,
        "refusal_reason": None,
        "trace": trace_event("guardrail", started, "stub: allowed"),
    }
