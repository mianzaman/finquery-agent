"""Entities node: resolves companies and periods without an LLM.

Companies come from two places: the names the guardrail's call listed, and a scan of the
question for known names, aliases and tickers (so a name the model missed still counts).
Each name is matched exactly or as a near miss; one that matches nothing ends the run with a
refusal that offers the closest companies. Time phrases are then resolved to explicit fiscal
periods for each company (see `app/agent/periods.py`).
"""

import time

from app.agent.companies import (
    CompanyIndex,
    build_index,
    describe,
    is_generic,
    is_sector,
    load_aliases,
    resolve_mention,
    scan_question,
)
from app.agent.periods import resolve_periods
from app.agent.state import AgentState, trace_event
from app.db import reference

MAX_QUOTED_NAME = 60  # characters of an unknown name repeated back to the user


def resolve_companies(
    question: str, mentions: list[str], index: CompanyIndex
) -> tuple[list[str], dict[str, list[str]]]:
    """Return the tickers found and, for each name that matched nothing, the closest tickers."""
    tickers: list[str] = []
    unknown: dict[str, list[str]] = {}
    for mention in mentions:
        if not isinstance(mention, str) or not mention.strip():
            continue
        if is_sector(mention, index) or is_generic(mention):
            continue  # a group, an index or a generic word: nothing to resolve, nothing to refuse
        ticker, suggestions = resolve_mention(mention, index)
        if ticker is None:
            unknown[mention] = suggestions
        else:
            tickers.append(ticker)
    tickers += scan_question(question, index)
    return list(dict.fromkeys(tickers)), unknown


def unknown_company_message(unknown: dict[str, list[str]], index: CompanyIndex) -> str:
    """The refusal shown when a company is not in the dataset."""
    parts: list[str] = []
    for mention, suggestions in unknown.items():
        # The name is the model's or the user's text: keep it short and on one line.
        name = " ".join(mention.split())[:MAX_QUOTED_NAME].replace('"', "'")
        hint = f" Did you mean {describe(suggestions, index)}?" if suggestions else ""
        parts.append(f'I could not find "{name}" among the companies I have data for.{hint}')
    covered = f"The data covers {len(index.tickers)} large US companies."
    return " ".join([*parts, covered])


def entities(state: AgentState) -> dict:
    """Resolve the question's companies to tickers and its time phrases to fiscal periods."""
    started = time.perf_counter()
    question = state.get("question", "")
    index = build_index(reference.companies(), load_aliases())
    tickers, unknown = resolve_companies(question, state.get("company_mentions") or [], index)
    if unknown:
        return {
            "refused": True,
            "refusal_reason": unknown_company_message(unknown, index),
            "tickers": tickers,
            "periods": [],
            "trace": trace_event("entities", started, f"unknown company: {', '.join(unknown)}"),
        }
    periods, notes = resolve_periods(question, tickers, reference.fiscal_calendar())
    note = f"{len(tickers)} companies, {len(periods)} periods"
    if notes:
        note += f"; {'; '.join(notes)}"
    return {"tickers": tickers, "periods": periods, "trace": trace_event("entities", started, note)}
