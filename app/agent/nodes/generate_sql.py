"""Generate-SQL node: writes a query for the question.

One model call. The prompt holds what the earlier nodes found: the relevant tables with their
columns, similar questions with correct queries, and the companies and periods the question
means. The query is not trusted: the next node validates it before anything runs.
"""

import time

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.context import describe_companies, describe_periods, format_examples
from app.agent.prompts import GENERATE_SQL_PROMPT, SQL_RULES
from app.agent.state import AgentState, trace_event
from app.db import reference
from app.llm import structured


class GeneratedSql(BaseModel):
    """What the model returns: the query and the tables it says it used."""

    sql: str = ""
    tables_used: list[str] = Field(default_factory=list)


def prompt_context(state: AgentState) -> dict[str, str]:
    """The parts of the state that both SQL prompts show, as text."""
    names = {company.ticker: company.name for company in reference.companies()}
    first, last = reference.data_range()
    return {
        "question": state.get("question", ""),
        "schema_context": state.get("schema_context", ""),
        "examples": format_examples(state.get("examples") or []),
        "companies": describe_companies(state.get("tickers") or [], names),
        "periods": describe_periods(state.get("periods") or []),
        "first_date": f"{first:%B %Y}" if first else "an unknown date",
        "last_date": f"{last:%d %B %Y}" if last else "an unknown date",
        "rules": SQL_RULES,
    }


def generate_sql(state: AgentState) -> dict:
    """Ask the model for a query. A model failure is raised, not stored."""
    started = time.perf_counter()
    prompt = GENERATE_SQL_PROMPT.format(**prompt_context(state))
    generated = structured(GeneratedSql, [HumanMessage(prompt)])
    tables = ", ".join(generated.tables_used) or "none named"
    return {
        "sql": generated.sql.strip(),
        "trace": trace_event("generate_sql", started, f"tables: {tables}"),
    }
