from __future__ import annotations

from collections.abc import Sequence

from jev_why.faithfulness import faithfulness_plan, score_faithfulness


def _evaluate(plan: object, truth: Sequence[float]) -> list[float]:
    """Value each coalition as the sum of its members' true weights."""
    return [sum(truth[i] for i in coalition) for coalition in plan.coalitions]  # type: ignore[attr-defined]


def test_a_correct_ranking_beats_random_deletion() -> None:
    truth = [0.9, 0.7, 0.5, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    ranking = sorted(range(10), key=lambda i: -truth[i])
    plan = faithfulness_plan(ranking, 10, random_trials=20, seed=1)
    report = score_faithfulness(plan, _evaluate(plan, truth))

    assert report.lift > 0
    assert report.p_value <= 0.1
    assert report.credible
    assert "faithful" in report.verdict()


def test_a_shuffled_ranking_does_not_beat_random() -> None:
    """The control has to be able to fail, or it is not a control."""
    truth = [0.9, 0.7, 0.5, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
    misleading = [6, 7, 8, 9, 5, 4, 3, 2, 1, 0]
    plan = faithfulness_plan(misleading, 10, random_trials=20, seed=1)
    report = score_faithfulness(plan, _evaluate(plan, truth))

    assert report.lift < 0
    assert not report.credible
    assert "no better than" in report.verdict()


def test_sufficiency_is_low_when_the_top_spans_carry_the_decision() -> None:
    truth = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    plan = faithfulness_plan([0, 1, 2, 3, 4, 5], 6, random_trials=5, seed=0)
    report = score_faithfulness(plan, _evaluate(plan, truth))
    assert report.sufficiency < report.comprehensiveness


def test_a_sign_flipped_explanation_trips_the_counter_evidence_check() -> None:
    """Nothing else in the metric set catches an explanation that has the
    direction backwards, which is the most dangerous way to be wrong."""
    truth = [0.8, 0.6, -0.9, -0.7, 0.0, 0.0]
    ranking = [0, 1, 2, 3, 4, 5]
    wrong_negatives = [0, 1]  # claims the positive spans are the negative ones
    plan = faithfulness_plan(ranking, 6, random_trials=5, negative_ranking=wrong_negatives, seed=0)
    report = score_faithfulness(plan, _evaluate(plan, truth))
    assert not report.counter_evidence_holds
    assert "sign" in report.verdict() or "negatively-scored" in report.verdict()


def test_correctly_identified_negative_spans_pass_the_check() -> None:
    truth = [0.8, 0.6, -0.9, -0.7, 0.0, 0.0]
    plan = faithfulness_plan(
        [2, 3, 0, 1, 4, 5], 6, random_trials=5, negative_ranking=[2, 3], seed=0
    )
    report = score_faithfulness(plan, _evaluate(plan, truth))
    assert report.counter_evidence_holds


def test_the_p_value_floor_is_set_by_the_number_of_trials() -> None:
    """With five trials the smallest achievable p is 1/6, so a report that
    quotes p must also say how many trials produced it."""
    truth = [1.0] + [0.0] * 9
    plan = faithfulness_plan(list(range(10)), 10, random_trials=5, seed=0)
    report = score_faithfulness(plan, _evaluate(plan, truth))
    assert report.random_trials == 5
    assert report.p_value >= 1 / 6


def test_the_schedule_never_deletes_everything_or_nothing() -> None:
    plan = faithfulness_plan(list(range(20)), 20, schedule=(0.0001, 0.5, 0.99, 1.5))
    assert all(1 <= k <= 19 for k in plan.ks)


def test_planning_deduplicates_coalitions_across_trials() -> None:
    plan = faithfulness_plan(list(range(8)), 8, random_trials=30, seed=3)
    assert len(set(plan.coalitions)) == plan.n_calls


def test_the_curve_reports_every_budget_point() -> None:
    truth = [0.5] * 6
    plan = faithfulness_plan(list(range(6)), 6, random_trials=3, seed=0)
    report = score_faithfulness(plan, _evaluate(plan, truth))
    assert len(report.curve) == len(plan.ks)
    assert all(0 < point.fraction < 1 for point in report.curve)


def test_cross_masking_is_recorded_on_the_report() -> None:
    """Scoring with the same masker used for attribution partly rewards the
    masker's own artifacts, so which mode was used has to travel with the
    number."""
    plan = faithfulness_plan(list(range(6)), 6, random_trials=3, seed=0)
    report = score_faithfulness(plan, _evaluate(plan, [0.1] * 6), cross_masked=True)
    assert report.cross_masked
