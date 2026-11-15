"""Repair-SQL node: fixes a query that failed validation or execution.

One model call with the question, the query that failed and the exact error, plus the same
tables, companies and periods the first attempt saw. The graph sends the result back to
validation, and stops asking for repairs after `MAX_REPAIRS`.
"""

import time

from langchain_core.messages import HumanMessage

from app.agent.nodes.generate_sql import GeneratedSql, prompt_context
from app.agent.prompts import REPAIR_SQL_PROMPT
from app.agent.state import AgentState, trace_event
from app.llm import structured


def repair_sql(state: AgentState) -> dict:
    """Ask the model to correct the failed query, and count the attempt."""
    started = time.perf_counter()
    error = state.get("validation_error") or state.get("db_error") or "the query failed"
    context = prompt_context(state)
    prompt = REPAIR_SQL_PROMPT.format(
        question=context["question"],
        sql=state.get("sql") or "(no query was written)",
        error=error,
        schema_context=context["schema_context"],
        companies=context["companies"],
        periods=context["periods"],
        rules=context["rules"],
    )
    repaired = structured(GeneratedSql, [HumanMessage(prompt)])
    attempts = state.get("attempts", 0) + 1
    return {
        "sql": repaired.sql.strip(),
        "validation_error": None,
        "db_error": None,
        "attempts": attempts,
        "trace": trace_event("repair_sql", started, f"repair {attempts} after: {error[:120]}"),
    }
