from __future__ import annotations

import pytest

from jev_why.questions import (
    Choice,
    Noul,
    QuestionError,
    Score,
    panel_payload,
    validate_panel,
)


def test_choice_rejects_more_options_than_the_api_allows() -> None:
    spec = Choice(instructions="pick", criteria={f"o{i}": None for i in range(256)})
    with pytest.raises(QuestionError, match="at most 255"):
        spec.validate("q")


def test_choice_needs_at_least_two_options_to_be_a_decision() -> None:
    with pytest.raises(QuestionError, match="at least 2 options"):
        Choice(instructions="pick", criteria={"only": None}).validate("q")


def test_score_levels_are_bounded_at_both_ends() -> None:
    with pytest.raises(QuestionError, match="between 2 and 10"):
        Score(instructions="rate", criteria=["one"]).validate("q")
    with pytest.raises(QuestionError, match="between 2 and 10"):
        Score(instructions="rate", criteria=[f"l{i}" for i in range(11)]).validate("q")


def test_empty_instructions_are_rejected_before_they_cost_a_call() -> None:
    with pytest.raises(QuestionError, match="instructions must not be empty"):
        Noul(instructions="   ").validate("q")


def test_noul_criteria_are_limited_to_true_and_false() -> None:
    with pytest.raises(QuestionError, match="only contain"):
        Noul(instructions="is it?", criteria={"maybe": "hmm"}).validate("q")


def test_validating_a_panel_catches_the_offending_question_by_name() -> None:
    panel = {"good": Noul(instructions="fine"), "bad": Noul(instructions="")}
    with pytest.raises(QuestionError, match="'bad'"):
        validate_panel(panel)


def test_an_empty_panel_is_an_error() -> None:
    with pytest.raises(QuestionError, match="at least one question"):
        validate_panel({})


def test_payload_is_key_sorted_so_the_cache_hits_across_declaration_orders() -> None:
    a = {"zebra": Noul(instructions="z"), "apple": Noul(instructions="a")}
    b = {"apple": Noul(instructions="a"), "zebra": Noul(instructions="z")}
    assert list(panel_payload(a)) == ["apple", "zebra"]
    assert panel_payload(a) == panel_payload(b)


def test_payloads_match_the_documented_request_shape() -> None:
    payload = panel_payload(
        {
            "urgent": Noul(instructions="is it urgent?"),
            "dept": Choice(instructions="which team?", criteria={"billing": "money", "tech": None}),
            "mood": Score(instructions="how annoyed?", criteria=["calm", "annoyed", "angry"]),
        }
    )
    assert payload["urgent"] == {"type": "noul", "instructions": "is it urgent?"}
    assert payload["dept"]["criteria"] == {"billing": "money", "tech": None}
    assert payload["mood"]["criteria"] == ["calm", "annoyed", "angry"]
