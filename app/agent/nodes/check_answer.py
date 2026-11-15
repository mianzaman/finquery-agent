"""Check-answer node: finds the numbers in the answer that the rows do not support.

No model. It stores the list; the graph decides what an unsupported number leads to (one more
answer, then the neutral sentence). The rules are in `app/agent/faithfulness.py`.
"""

import time

from app.agent.faithfulness import unsupported_numbers
from app.agent.nodes.generate_sql import prompt_context
from app.agent.state import AgentState, trace_event


def allowed_text(state: AgentState) -> str:
    """Text whose numbers an answer may repeat: the question and what it was resolved to."""
    context = prompt_context(state)
    return "\n".join((context["question"], context["companies"], context["periods"]))


def check_answer(state: AgentState) -> dict:
    """Store the answer's unsupported numbers; an empty list means the answer is faithful."""
    started = time.perf_counter()
    missing = unsupported_numbers(
        state.get("answer", ""), state.get("rows") or [], allowed_text(state)
    )
    note = f"unsupported: {', '.join(missing)}"[:160] if missing else "every number is supported"
    return {
        "unsupported_numbers": missing,
        "faithfulness_failed": False,
        "trace": trace_event("check_answer", started, note),
    }
