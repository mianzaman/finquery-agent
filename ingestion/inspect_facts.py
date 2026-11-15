"""Prints summaries of a cached SEC company-facts file. It never prints a whole file.

uv run python -m ingestion.inspect_facts XOM                     # tags the company uses
uv run python -m ingestion.inspect_facts XOM --match Revenue     # only tags containing a word
uv run python -m ingestion.inspect_facts XOM --tag Revenues      # periods and forms of one tag
uv run python -m ingestion.inspect_facts XOM --filings           # one line per 10-K and 10-Q
"""

import argparse
import json
import sys
from collections import Counter
from typing import Any

from ingestion.load_sec import RAW_DIR, read_mapping

MAX_LINES = 400


def load_facts(ticker: str) -> dict[str, Any]:
    """Read one cached file; the loader must have downloaded it already."""
    path = RAW_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        raise SystemExit(f"no cache file for {ticker}; run ingestion.load_sec first")
    return json.loads(path.read_text(encoding="utf-8"))


def tag_rows(facts: dict[str, Any], tag: str) -> list[tuple[str, dict[str, Any]]]:
    """Every fact of a tag as `(unit, fact)`, across taxonomies."""
    found = []
    for taxonomy in facts.get("facts", {}).values():
        for unit, rows in taxonomy.get(tag, {}).get("units", {}).items():
            found += [(unit, row) for row in rows]
    return found


def summarise_tags(facts: dict[str, Any], match: str | None, since: str) -> list[str]:
    """One line per tag: fact count, forms, and the range of period ends."""
    lines = []
    for taxonomy, tags in facts.get("facts", {}).items():
        for tag, body in sorted(tags.items()):
            if match and match.lower() not in tag.lower():
                continue
            rows = [r for unit in body.get("units", {}).values() for r in unit]
            recent = [r for r in rows if r["end"] >= since]
            if not recent:
                continue
            forms = Counter(r.get("form") for r in recent)
            lines.append(
                f"{taxonomy}:{tag}  facts={len(recent)}  "
                f"ends {min(r['end'] for r in recent)}..{max(r['end'] for r in recent)}  "
                f"forms={dict(forms.most_common(4))}"
            )
    return lines


def summarise_tag(facts: dict[str, Any], tag: str, since: str) -> list[str]:
    """One line per fact of a tag: period, value, form, fiscal labels and filing."""
    rows = [(unit, r) for unit, r in tag_rows(facts, tag) if r["end"] >= since]
    rows.sort(key=lambda item: (item[1]["end"], item[1].get("start", ""), item[1].get("filed", "")))
    return [
        f"{r.get('start', '          ')} -> {r['end']}  {r['val']:>18}  {unit:<10} "
        f"{r.get('form', '?'):<7} fy={r.get('fy')} fp={r.get('fp')}  filed {r.get('filed')}  "
        f"{r.get('accn')}"
        for unit, r in rows
    ]


def summarise_filings(facts: dict[str, Any], since: str) -> list[str]:
    """One line per filing: its form, labels, and the period ends the mapped tags carry."""
    mapping = read_mapping()
    tags = [
        tag
        for columns in mapping.values()
        if isinstance(columns, dict)
        for candidates in columns.values()
        for tag in candidates
    ]
    filings: dict[str, dict[str, Any]] = {}
    for tag in tags:
        for _, row in tag_rows(facts, tag):
            entry = filings.setdefault(
                row.get("accn", "?"),
                {
                    "form": row.get("form"),
                    "fy": row.get("fy"),
                    "fp": row.get("fp"),
                    "filed": row.get("filed"),
                    "ends": Counter(),
                },
            )
            entry["ends"][row["end"]] += 1
    lines = []
    for accn, entry in sorted(filings.items(), key=lambda item: item[1]["filed"] or ""):
        if (entry["filed"] or "") < since:
            continue
        ends = ", ".join(f"{end} x{n}" for end, n in sorted(entry["ends"].items())[-4:])
        lines.append(
            f"{entry['filed']}  {entry['form'] or '?':<7} fy={entry['fy']} fp={entry['fp']}  "
            f"{accn}  latest ends: {ends}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Summarise a cached SEC company-facts file.")
    parser.add_argument("ticker")
    parser.add_argument("--match", help="only tags whose name contains this text")
    parser.add_argument("--tag", help="show the periods and forms of this one tag")
    parser.add_argument("--filings", action="store_true", help="one line per filing")
    parser.add_argument("--since", default="2020-01-01", help="ignore anything earlier (ISO date)")
    args = parser.parse_args(argv)

    facts = load_facts(args.ticker)
    if args.tag:
        lines = summarise_tag(facts, args.tag, args.since)
    elif args.filings:
        lines = summarise_filings(facts, args.since)
    else:
        lines = summarise_tags(facts, args.match, args.since)

    print(f"{args.ticker.upper()}: {facts.get('entityName', '?')}  ({len(lines)} lines)")
    for line in lines[:MAX_LINES]:
        print(line)
    if len(lines) > MAX_LINES:
        print(f"... {len(lines) - MAX_LINES} more lines; narrow with --match or --since")
    return 0


if __name__ == "__main__":
    sys.exit(main())
