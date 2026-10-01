"""Frozen revision-v2 answer scoring and response diagnostics.

Primary rule (the v1 rule): the last ####-marked integer or decimal anywhere in
the response, with no fallback. Two sensitivity rules are computed for every
response: the first marked answer, and the last marked answer with a fallback
to the last number anywhere in the response. Numbers are compared as exact
finite decimals after removing commas, so 8.0 equals 8.

This module is frozen by the tag protocol-v2-frozen. Its SHA-256 is recorded
in every evaluation manifest. Do not change it after the freeze.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation

SCORER_VERSION = "revision-v2-scorer-1.0"
PRIMARY_RULE = "last_marked"
NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MARKED = re.compile(rf"####\s*({NUMBER})")
ANY_NUMBER = re.compile(rf"(?<![\d.,])({NUMBER})")
MARKED_LINE = re.compile(rf"^####\s*{NUMBER}\s*[.!?]?$")
LOOP_WINDOW = 20
LOOP_MIN_REPEATS = 5


def normalize_number(value: str) -> str | None:
    """Return a canonical finite decimal string, or None if the value is not one."""
    compact = value.strip().replace(",", "")
    try:
        number = Decimal(compact)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def last_marked(text: str) -> str | None:
    """Primary rule: last ####-marked number, no fallback."""
    matches = MARKED.findall(text)
    return normalize_number(matches[-1]) if matches else None


def first_marked(text: str) -> str | None:
    """Sensitivity rule: first ####-marked number, no fallback."""
    matches = MARKED.findall(text)
    return normalize_number(matches[0]) if matches else None


def last_marked_fallback_last_number(text: str) -> str | None:
    """Sensitivity rule: last marked number, else the last number anywhere."""
    if MARKED.search(text):
        return last_marked(text)
    numbers = ANY_NUMBER.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


RULES: dict[str, Callable[[str], str | None]] = {
    "last_marked": last_marked,
    "first_marked": first_marked,
    "last_marked_fallback_last_number": last_marked_fallback_last_number,
}


def answers_equal(predicted: str | None, reference: str | None) -> bool:
    """Exact finite-decimal comparison; a missing answer is never correct."""
    if predicted is None or reference is None:
        return False
    try:
        return Decimal(predicted) == Decimal(reference)
    except InvalidOperation:
        return False


def majority_vote(
    answers: Sequence[str | None], *, greedy_answer: str | None = None
) -> tuple[str | None, bool]:
    """Most frequent valid answer; a tie goes to the greedy answer, then numeric, then lexical order."""
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return None, False
    counts = Counter(valid)
    highest = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == highest]
    tied = len(winners) > 1
    if greedy_answer in winners:
        return greedy_answer, tied
    winners.sort(key=lambda answer: (Decimal(answer), answer))
    return winners[0], tied


def non_blank_lines(text: str) -> list[str]:
    """Return stripped non-blank lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_bare_answer(text: str) -> bool:
    """At least one ####-marked line and no other non-blank text."""
    lines = non_blank_lines(text)
    return bool(lines) and all(MARKED_LINE.match(line) for line in lines)


def is_repetition_loop(text: str) -> bool:
    """One line occurs at least 5 times and in at least half of the last 20 non-blank lines."""
    tail = non_blank_lines(text)[-LOOP_WINDOW:]
    if not tail:
        return False
    count = Counter(tail).most_common(1)[0][1]
    return count >= LOOP_MIN_REPEATS and count * 2 >= len(tail)


def has_question_line(text: str) -> bool:
    """At least one non-blank line ends in a question mark."""
    return any(line.endswith("?") for line in non_blank_lines(text))


def diagnose(text: str, finish_reason: str | None) -> dict[str, bool]:
    """Per-response format diagnostics defined in the protocol."""
    return {
        "bare_answer": is_bare_answer(text),
        "limit_hit": finish_reason == "length",
        "repetition_loop": is_repetition_loop(text),
        "question_lines": has_question_line(text),
    }


def extract_all(text: str) -> dict[str, str | None]:
    """Apply every frozen rule to one response."""
    return {name: rule(text) for name, rule in RULES.items()}
