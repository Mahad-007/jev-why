from __future__ import annotations

import pytest

from jev_why.types import Answer
from jev_why.values import (
    ValueSpec,
    choose_link,
    expected_level,
    is_saturated,
    jensen_shannon,
    logit,
    magnitude,
    primary_value,
    wasserstein1,
)


def test_saturated_probabilities_switch_to_log_odds() -> None:
    """At p=0.995 no span can move the probability by more than 0.005, so
    ranking in probability space collapses into noise."""
    assert is_saturated(0.99) and is_saturated(0.01)
    assert not is_saturated(0.5)
    assert choose_link(0.99, "auto") == "logit"
    assert choose_link(0.5, "auto") == "prob"
    assert choose_link(0.99, "prob") == "prob", "an explicit request wins"


def test_logit_clips_rather_than_diverging() -> None:
    assert logit(0.0) < 0 and logit(1.0) > 0
    assert abs(logit(0.5)) < 1e-12


def test_score_value_is_recomputed_from_the_distribution() -> None:
    """E is linear in the answer distribution, so Shapley over E decomposes
    exactly into Shapley over each per-level probability."""
    answer = Answer(qtype="score", score=9.9, probabilities={0: 0.5, 1: 0.5})
    assert primary_value(answer, ValueSpec("f", "score")) == pytest.approx(0.5)


def test_score_falls_back_to_the_scalar_when_no_distribution() -> None:
    answer = Answer(qtype="score", score=7.25)
    assert primary_value(answer, ValueSpec("f", "score")) == pytest.approx(7.25)


def test_expected_level_normalises_an_unnormalised_distribution() -> None:
    assert expected_level({0: 1.0, 2: 1.0}) == pytest.approx(1.0)


def test_wasserstein_respects_level_ordering_where_jsd_cannot() -> None:
    """The reason score magnitude is not Jensen-Shannon.

    Moving the same mass one level and three levels are different events. An
    ordinal metric must rank the three-level move higher in proportion; JSD
    scores them almost identically because it only sees that mass moved between
    two distinct categories.
    """
    base = (0.4, 0.2, 0.2, 0.2)
    near = (0.1, 0.5, 0.2, 0.2)  # 0.3 moved from level 0 to level 1
    far = (0.1, 0.2, 0.2, 0.5)  # 0.3 moved from level 0 to level 3

    w_near, w_far = wasserstein1(base, near), wasserstein1(base, far)
    j_near, j_far = jensen_shannon(base, near), jensen_shannon(base, far)

    assert w_far == pytest.approx(3 * w_near, rel=1e-9)
    assert j_far == pytest.approx(j_near, rel=0.05), "JSD is blind to the ordering"


def test_choice_magnitude_sees_reshuffling_that_leaves_the_top_class_alone() -> None:
    base = Answer(qtype="choice", choice="a", probabilities={"a": 0.5, "b": 0.4, "c": 0.1})
    shuffled = Answer(qtype="choice", choice="a", probabilities={"a": 0.5, "b": 0.1, "c": 0.4})
    assert primary_value(base, ValueSpec("f", "choice", target="a")) == pytest.approx(
        primary_value(shuffled, ValueSpec("f", "choice", target="a"))
    )
    assert magnitude(base, shuffled, "choice") > 0.05


def test_contrast_explains_a_pairwise_decision() -> None:
    answer = Answer(
        qtype="choice",
        choice="billing",
        probabilities={"billing": 0.6, "technical": 0.3, "sales": 0.1},
    )
    spec = ValueSpec("dept", "choice", contrast=("billing", "technical"))
    assert primary_value(answer, spec) == pytest.approx(logit(0.6) - logit(0.3))


def test_choice_target_defaults_to_the_selected_class() -> None:
    answer = Answer(
        qtype="choice", choice="billing", probabilities={"billing": 0.7, "technical": 0.3}
    )
    assert primary_value(answer, ValueSpec("d", "choice")) == pytest.approx(0.7)


def test_jensen_shannon_is_bounded_and_zero_on_identity() -> None:
    assert jensen_shannon((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)
    assert 0.99 < jensen_shannon((1.0, 0.0), (0.0, 1.0)) <= 1.0


def test_noul_magnitude_is_absolute_probability_change() -> None:
    a = Answer(qtype="noul", noul=0.9)
    b = Answer(qtype="noul", noul=0.4)
    assert magnitude(a, b, "noul") == pytest.approx(0.5)
