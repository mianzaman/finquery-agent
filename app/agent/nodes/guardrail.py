"""Guardrail node: refuses out-of-scope questions.

One model call decides whether the question can be answered from the data, and in the same
call names the kind of question and the companies it mentions, so the next two nodes need no
call of their own. Obvious attempts to change data are refused before that, with no call.

A refusal always means "out of scope". If the model cannot be reached, the error is raised
(`LLMError`, or `RateLimitedError` for a rate limit) instead of being reported as a refusal.
"""

import re
import time

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.prompts import GUARDRAIL_PROMPT
from app.agent.state import AgentState, trace_event
from app.db import reference
from app.llm import structured

# SQL commands that write. Only the command itself matches, so ordinary sentences ("did the
# price drop", "delete any guidance", "insert into my watchlist") go on to the model, which
# also refuses plain-English requests to change data ("delete all rows from ...").
WRITE_ATTEMPT = re.compile(
    r"\bdrop\s+(?:table|schema|database)\b"
    r"|\bdelete\s+from\b"
    r"|\binsert\s+into\s+\w+\s*(?:\(|values\b|select\b)"
    r"|\btruncate\b"
    r"|\bupdate\s+\w+\s+set\b"
    r"|\balter\s+table\b",
    re.IGNORECASE,
)
MAX_QUESTION_LENGTH = 1000  # characters; bounds the model's input and the rule-based matching
MAX_MENTIONS = 20
MAX_MENTION_LENGTH = 100
LONG_REFUSAL = f"Please ask a shorter question (at most {MAX_QUESTION_LENGTH} characters)."
WRITE_REFUSAL = "I can only read the data to answer questions; I cannot change or delete it."
EMPTY_REFUSAL = "Please ask a question about a company's financials, share price or dividends."
DEFAULT_REFUSAL = "That question cannot be answered from the financial data I have."


class GuardrailVerdict(BaseModel):
    """What the model returns for one question."""

    allowed: bool
    reason: str = ""
    # Free text on purpose: an unexpected label must not fail the call. The intent node checks it.
    intent: str | None = None
    companies: list[str] = Field(default_factory=list)


def _refuse(reason: str, started: float, note: str) -> dict:
    return {
        "refused": True,
        "refusal_reason": reason,
        "intent": None,
        "company_mentions": [],
        "trace": trace_event("guardrail", started, note),
    }


def guardrail(state: AgentState) -> dict:
    """Refuse an out-of-scope question, or pass it on with its intent and company mentions."""
    started = time.perf_counter()
    question = state.get("question", "").strip()
    if not question:
        return _refuse(EMPTY_REFUSAL, started, "refused without a model call: empty question")
    if len(question) > MAX_QUESTION_LENGTH:
        return _refuse(LONG_REFUSAL, started, "refused without a model call: too long")
    if WRITE_ATTEMPT.search(question):
        return _refuse(WRITE_REFUSAL, started, "refused without a model call: write attempt")

    first, last = reference.data_range()
    prompt = GUARDRAIL_PROMPT.format(
        question=question,
        first_date=f"{first:%B %Y}" if first else "an unknown date",
        last_date=f"{last:%d %B %Y}" if last else "an unknown date",
    )
    verdict = structured(GuardrailVerdict, [HumanMessage(prompt)])
    if not verdict.allowed:
        return _refuse(verdict.reason.strip() or DEFAULT_REFUSAL, started, "refused by the model")
    # The list is the model's, so it is bounded before any matching is done on it.
    mentions = [name.strip() for name in verdict.companies if name and name.strip()]
    mentions = [name for name in mentions if len(name) <= MAX_MENTION_LENGTH][:MAX_MENTIONS]
    return {
        "refused": False,
        "refusal_reason": None,
        "intent": verdict.intent,
        "company_mentions": mentions,
        "trace": trace_event("guardrail", started, f"allowed; {len(mentions)} companies named"),
    }
