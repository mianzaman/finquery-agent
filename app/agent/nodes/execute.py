"""Execute node: runs validated SQL on the read-only engine.

It runs `state["sql"]` exactly as the validation node stored it, through `run_select`: the
`agent_reader` role, a read-only transaction, a 5 second timeout and a single statement. A
failure becomes `db_error` (the database's own message, first line), which the repair node reads.
"""

import time

from app.agent.nodes.validate_sql import MAX_LIMIT
from app.agent.state import AgentState, trace_event
from app.db.session import QueryError, run_select


def execute(state: AgentState) -> dict:
    """Run the query and store its columns and rows, or the database's error."""
    started = time.perf_counter()
    sql = state.get("sql")
    if not sql:
        error = "there is no query to run"
        return {"db_error": error, "trace": trace_event("execute", started, error)}
    try:
        columns, rows, _ = run_select(sql, max_rows=MAX_LIMIT)
    except QueryError as exc:
        return {"db_error": str(exc), "trace": trace_event("execute", started, f"failed: {exc}")}
    return {
        "columns": columns,
        "rows": rows,
        "db_error": None,
        "trace": trace_event("execute", started, f"{len(rows)} rows"),
    }
