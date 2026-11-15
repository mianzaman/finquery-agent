"""Tests for the intent node: no model call, only a check of the guardrail's label."""

from typing import Any

import pytest

from app.agent.nodes.intent import intent
from app.agent.state import INTENTS


@pytest.mark.parametrize("label", INTENTS)
def test_a_known_intent_is_kept(label: str) -> None:
    update = intent({"intent": label})

    assert set(update) == {"intent", "trace"}
    assert update["intent"] == label
    assert update["trace"][0]["note"] == label


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Ranking", "ranking"),
        (" trend ", "trend"),
        ("multi-table", "multi_table"),
        ("Multi Table", "multi_table"),
    ],
)
def test_spelling_variants_of_a_known_intent_are_accepted(given: str, expected: str) -> None:
    assert intent({"intent": given})["intent"] == expected


@pytest.mark.parametrize(
    ("state", "note"),
    [
        ({}, "no intent from the guardrail; using lookup"),
        ({"intent": None}, "no intent from the guardrail; using lookup"),
        ({"intent": ""}, "no intent from the guardrail; using lookup"),
        ({"intent": "time series"}, "unknown intent 'time series'; using lookup"),
        ({"intent": "out_of_scope"}, "unknown intent 'out_of_scope'; using lookup"),
    ],
)
def test_a_missing_or_unknown_intent_becomes_lookup_and_is_noted(
    state: dict[str, Any], note: str
) -> None:
    update = intent(state)

    assert update["intent"] == "lookup"
    assert update["trace"][0]["note"] == note
