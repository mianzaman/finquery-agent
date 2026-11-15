"""Faithfulness check: does every number in an answer come from the query's rows?

Rules only, no model and no database. `extract_numbers` reads the figures out of a sentence
(`$1.2B`, `24.5%`, `1,234`, `-5`); `unsupported_numbers` returns the ones it cannot trace to

- a number in the rows, within the rounding the answer itself shows (`$1.2B` is 1.15 to 1.25
  billion), or 100 times it when the answer writes a percentage of a stored fraction;
- a number inside a text cell of the rows;
- for a bare number only (no currency sign, scale or %): a year, month or day of a date in
  the rows, or the row count;
- a number in the text the caller allows (the question, the companies and periods it resolved),
  so "top 5" and "fiscal 2025" are not treated as claims about the data;
- a figure worked out from two numbers in the rows: the difference of two values in one column
  (a change between periods, a gap between companies), or the ratio or percentage change of
  two values in one column or in one row (growth, "1.5 times", a margin).

Values from different columns of different rows are never combined: with enough unrelated
numbers almost any figure equals some difference or ratio.

What it does not catch, by design: a number given to the wrong company or period, a loss
written without its sign, and a wrong figure that happens to equal one of the worked-out
values.

The work is bounded whatever the answer holds, because the answer is model output: worked-out
values are built once per call and searched by bisection, they are not tried for large
results, and only the first `MAX_CLAIMS` distinct numbers are examined (the rest are reported
as unsupported).
"""

import math
import re
from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import permutations
from typing import Any

MAX_CLAIMS = 50  # distinct numbers examined in one answer; 1 to 3 sentences hold far fewer
MAX_DERIVED_VALUES = 200  # above this many numeric cells, worked-out values are not tried
RELATIVE_SLACK = 1e-12  # float noise, on top of the rounding the answer shows

SCALES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NUMBER = re.compile(
    r"""
    (?<![\w.,])                                   # not inside a word ("Q3") or another number
    (?P<open>\(\s?(?=-?[$€£]))?                   # "($5)": accounting style for a negative
    (?P<sign>(?<![\w)])[-−](?=[$€£]?\.?\d))?      # a minus, but not the hyphen of "2021-2025"
    (?P<currency>[$€£])?
    (?P<inner_sign>[-−])?                         # "$-5"
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?:
        (?P<letter>[KkMBT]|bn|mn)(?![A-Za-z])     # "1.2B", "3.4M", "12bn"
      | \s?(?P<word>(?i:thousand|million|billion|trillion))\b
    )?
    (?P<percent>\s?(?:%|(?i:percent\b|percentage\spoints?\b)))?
    (?P<close>\))?
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Claim:
    """One number as an answer wrote it."""

    text: str  # as written, for the report: "$1.2B"
    value: float  # what it says, sign dropped: 1_200_000_000.0
    step: float  # the size of its last written digit: 100_000_000.0
    percent: bool = False
    negative: bool = False
    bare: bool = True  # written with no currency sign, scale or percent sign

    @property
    def tolerance(self) -> float:
        """Half the last written digit: what rounding to that digit can hide."""
        return self.step / 2


def _scale(match: re.Match[str]) -> float:
    letter, word = match.group("letter"), match.group("word")
    if word:
        return SCALES[word.lower()]
    # Only "K", "k", "M", "B", "T", "bn" and "mn" get this far: a lower-case "m", "b" or "t"
    # after a number is more often a unit than a scale, so the pattern leaves it alone.
    return SCALES[letter.lower()] if letter else 1.0


def extract_numbers(text: str) -> list[Claim]:
    """Every number in `text`, in order. ISO dates count as a year, a month and a day."""
    spaced = _ISO_DATE.sub(r"\1 \2 \3", text)
    claims: list[Claim] = []
    for match in _NUMBER.finditer(spaced):
        digits = match.group("digits")
        try:
            number = Decimal(digits.replace(",", ""))
        except InvalidOperation:
            continue
        scale = _scale(match)
        decimals = len(digits.partition(".")[2])
        accounting = bool(match.group("open") and match.group("close"))
        negative = bool(match.group("sign") or match.group("inner_sign")) or accounting
        written = match.group(0)
        if match.group("close") and not accounting:
            written = written[:-1]  # the bracket closes the sentence's own "(", not the number
        claims.append(
            Claim(
                text=written.strip(),
                value=float(number) * scale,
                step=10.0**-decimals * scale,
                percent=bool(match.group("percent")),
                negative=negative,
                bare=not (match.group("currency") or scale != 1.0 or match.group("percent")),
            )
        )
    return claims


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    return math.isfinite(value)


def _numeric_cells(rows: Sequence[Sequence[Any]]) -> list[tuple[int, int, float]]:
    """Every numeric cell as (row index, column index, value)."""
    return [
        (r, c, float(value))
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
        if _is_number(value)
    ]


def _text_numbers(rows: Sequence[Sequence[Any]], allowed_text: str) -> set[float]:
    """Numbers written in the allowed text or in the rows' text cells: labels, not figures."""
    numbers = {claim.value for claim in extract_numbers(allowed_text)}
    for row in rows:
        for value in row:
            if isinstance(value, str):
                numbers |= {claim.value for claim in extract_numbers(value)}
    return numbers


def _calendar_numbers(rows: Sequence[Sequence[Any]]) -> set[float]:
    """The row count and the parts of the rows' dates, which only a bare number may repeat."""
    numbers = {float(len(rows))}
    for row in rows:
        for value in row:
            if isinstance(value, date):  # a datetime is a date too
                numbers |= {float(value.year), float(value.month), float(value.day)}
    return numbers


def _close(claim: Claim, value: float) -> bool:
    return abs(claim.value - abs(value)) <= claim.tolerance + RELATIVE_SLACK * abs(value)


def _matches_value(claim: Claim, value: float) -> bool:
    """True when the claim is `value` as written, or its percentage form."""
    if claim.negative and value >= 0:
        return False
    return _close(claim, value) or (claim.percent and _close(claim, value * 100))


def _pairs(cells: list[tuple[int, int, float]]) -> Iterator[tuple[float, float, bool]]:
    """Ordered pairs of cells that share a column or a row, with whether it is a column."""
    by_column: dict[int, list[float]] = {}
    by_row: dict[int, list[float]] = {}
    for row, column, value in cells:
        by_column.setdefault(column, []).append(value)
        by_row.setdefault(row, []).append(value)
    for values in by_column.values():
        for first, second in permutations(dict.fromkeys(values), 2):
            yield first, second, True
    for values in by_row.values():
        for first, second in permutations(values, 2):
            yield first, second, False


@dataclass(frozen=True)
class Derived:
    """Sizes of the values that two related cells give, sorted, for plain and percent claims."""

    plain: list[float]
    percent: list[float]

    def holds(self, claim: Claim) -> bool:
        values = self.percent if claim.percent else self.plain
        slack = claim.tolerance + RELATIVE_SLACK * claim.value
        index = bisect_left(values, claim.value - slack)
        return index < len(values) and values[index] <= claim.value + slack


def derive(cells: list[tuple[int, int, float]]) -> Derived:
    """Differences, ratios and percentage changes of every pair of related cells."""
    plain: list[float] = []
    percent: list[float] = []
    if len(cells) > MAX_DERIVED_VALUES:
        return Derived(plain, percent)
    for first, second, same_column in _pairs(cells):
        if same_column:
            difference = abs(first - second)
            plain.append(difference)
            # Two stored fractions (0.46 and 0.69) differ by 23 percentage points.
            percent += (difference, difference * 100)
        if second == 0:
            continue
        ratio = first / second
        plain.append(abs(ratio))
        percent += (abs(ratio * 100), abs((ratio - 1) * 100))
    return Derived(sorted(plain), sorted(percent))


def _supported(
    claim: Claim, cells: list[tuple[int, int, float]], labels: set[float], calendar: set[float]
) -> bool:
    """True when the claim is a row value or a label, without working anything out."""
    if any(_matches_value(claim, value) for _, _, value in cells):
        return True
    context = labels | calendar if claim.bare else labels
    return not claim.negative and any(_close(claim, value) for value in context)


def unsupported_numbers(
    answer: str, rows: Sequence[Sequence[Any]], allowed_text: str = ""
) -> list[str]:
    """The numbers in `answer`, as written, that cannot be traced to `rows` or `allowed_text`."""
    cells = _numeric_cells(rows)
    labels = _text_numbers(rows, allowed_text)
    calendar = _calendar_numbers(rows)
    claims = list({claim.text: claim for claim in extract_numbers(answer)}.values())
    doubtful = [c for c in claims[:MAX_CLAIMS] if not _supported(c, cells, labels, calendar)]
    derived = derive(cells) if doubtful else Derived([], [])
    unsupported = [claim for claim in doubtful if not derived.holds(claim)]
    return [claim.text for claim in (*unsupported, *claims[MAX_CLAIMS:])]
