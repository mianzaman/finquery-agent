"""Turns the resolved entities and retrieved examples into the text the SQL prompts show.

Pure formatting: tickers, periods and examples in, short lines out. Company names come from the
caller, so nothing here touches the database.
"""

from typing import Any

NO_COMPANIES = "none named (the question is about a group of companies, or all of them)"
NO_PERIODS = "- none resolved; take the period from the question's own wording"
NO_EXAMPLES = "(none)"
ASSUMED = " (the question did not say fiscal or calendar; the fiscal year is assumed)"

RELATIVE = {
    "latest_quarter": "the latest quarter in the data",
    "latest_year": "the latest complete fiscal year in the data",
    "year_to_date": "the fiscal year in progress, to date",
}


def describe_companies(tickers: list[str], names: dict[str, str]) -> str:
    """`MSFT (Microsoft), GOOGL (Alphabet)`, or a note that no company was named."""
    if not tickers:
        return NO_COMPANIES
    return ", ".join(f"{t} ({names[t]})" if t in names else t for t in tickers)


def _quarters(rows: list[dict[str, Any]]) -> str:
    """`fiscal 2025 Q1-Q4, fiscal 2026 Q1` for rows of one company, oldest first."""
    by_year: dict[int, list[int]] = {}
    for row in rows:
        by_year.setdefault(row["fiscal_year"], []).append(row["fiscal_quarter"])
    parts: list[str] = []
    for year, quarters in by_year.items():
        ordered = sorted(quarters)
        span = f"Q{ordered[0]}" if len(ordered) == 1 else f"Q{ordered[0]}-Q{ordered[-1]}"
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            span = ", ".join(f"Q{q}" for q in ordered)
        parts.append(f"fiscal {year} {span}")
    return ", ".join(parts)


def _meaning(row: dict[str, Any]) -> str:
    relative = row.get("relative") or ""
    if relative in RELATIVE:
        return RELATIVE[relative]
    if relative.startswith("last_") and relative.endswith("_quarters"):
        return f"the latest {relative.split('_')[1]} quarters in the data"
    if relative.startswith("since_fiscal_"):
        return f"fiscal {relative.rsplit('_', 1)[1]} onwards"
    if row.get("basis") == "calendar":
        year = (row.get("period_end") or "")[:4]
        return f"calendar year {year}"
    return ""


def describe_periods(periods: list[dict[str, Any]]) -> str:
    """One line per company and phrase, saying which fiscal quarters and dates it covers."""
    if not periods:
        return NO_PERIODS
    lines: list[str] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in periods:
        key = (row.get("ticker"), row.get("basis"), row.get("relative"), row.get("assumed"))
        groups.setdefault(key, []).append(row)
    for (ticker, basis, _, assumed), rows in groups.items():
        meaning = _meaning(rows[0])
        note = ASSUMED if assumed else ""
        if ticker is None:
            lines += [f"- {_neutral(row, meaning)}{note}" for row in rows]
            continue
        ends = [row["period_end"] for row in rows]
        dates = ends[0] if len(ends) == 1 else f"{min(ends)} to {max(ends)}"
        if basis == "calendar":
            year = ends[0][:4]
            lines.append(
                f"- {ticker}: {meaning}, dates {year}-01-01 to {year}-12-31 "
                f"(its quarters ending in that range: {_quarters(rows)})"
            )
            continue
        label = f"{meaning}: " if meaning else ""
        lines.append(f"- {ticker}: {label}{_quarters(rows)}; period_end {dates}{note}")
    return "\n".join(lines)


def _neutral(row: dict[str, Any], meaning: str) -> str:
    """A period that applies to whichever companies the query selects."""
    if row.get("basis") == "calendar":
        return f"each company: {meaning} ({row['period_start']} to {row['period_end']})"
    if meaning:
        return f"each company: {meaning} (its own newest rows, by period_end)"
    quarter = f" Q{row['fiscal_quarter']}" if row.get("fiscal_quarter") else ""
    return f"each company: fiscal {row['fiscal_year']}{quarter}"


def format_examples(examples: list[dict[str, str]]) -> str:
    """The retrieved examples as `Q:` / `SQL:` pairs."""
    if not examples:
        return NO_EXAMPLES
    return "\n\n".join(f"Q: {e['question']}\nSQL: {e['sql']}" for e in examples)
