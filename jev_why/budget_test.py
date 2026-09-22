from __future__ import annotations

import pytest

from jev_why.budget import (
    Budget,
    BudgetExceeded,
    BudgetGuard,
    amortisation_factor,
    estimate_cost,
    estimate_plan_cost,
)


def test_a_typical_explanation_costs_a_fraction_of_a_cent() -> None:
    """The claim the whole project rests on, asserted rather than quoted."""
    estimate = estimate_plan_cost(
        calls=82, state_tokens=500, question_tokens=200, mean_kept_fraction=0.5
    )
    assert estimate.usd < 0.01


def test_cost_grows_with_the_square_of_document_size() -> None:
    """Calls grow with span count, and span count grows with document length,
    so cost is quadratic. This is the limit that belongs in the README next to
    the fraction-of-a-cent figure."""
    small = estimate_plan_cost(calls=2 * 40 + 2, state_tokens=500, question_tokens=200)
    large = estimate_plan_cost(calls=2 * 400 + 2, state_tokens=5000, question_tokens=200)
    assert large.usd / small.usd > 50


def test_amortisation_is_large_but_not_unbounded() -> None:
    """Question specs are input tokens on every call, so sharing a coalition
    sample across a panel saves (K*S + Q)/(S + Q), not an infinite factor."""
    assert amortisation_factor(500, 200, 20) == pytest.approx((20 * 500 + 200) / 700)
    assert amortisation_factor(500, 200, 20) < 20


def test_amortisation_inverts_for_many_tiny_questions_on_a_short_state() -> None:
    """The case where the flattering assumption would be wrong."""
    assert amortisation_factor(state_tokens=100, question_tokens=4000, n_questions=200) < 200


def test_preflight_refuses_a_plan_that_would_blow_the_ceiling() -> None:
    guard = BudgetGuard(Budget(max_usd=0.0001))
    with pytest.raises(BudgetExceeded, match="would cost"):
        guard.preflight(calls=5000, state_tokens=50_000, question_tokens=500)


def test_preflight_message_says_what_to_do_about_it() -> None:
    guard = BudgetGuard(Budget(max_usd=1e-9))
    with pytest.raises(BudgetExceeded) as excinfo:
        guard.preflight(calls=100, state_tokens=1000, question_tokens=100)
    assert "max_spans" in str(excinfo.value)


def test_a_call_ceiling_is_enforced_separately_from_a_dollar_ceiling() -> None:
    guard = BudgetGuard(Budget(max_usd=None, max_calls=10))
    with pytest.raises(BudgetExceeded, match="needs 50 calls"):
        guard.preflight(calls=50, state_tokens=10, question_tokens=10)


def test_spending_is_enforced_at_runtime_not_only_in_preflight() -> None:
    """Pre-flight is an estimate. Actual usage is what the API reports."""
    guard = BudgetGuard(Budget(max_usd=estimate_cost(1000)))
    for _ in range(10):
        guard.record(input_tokens=200)
    with pytest.raises(BudgetExceeded, match="after 10 calls"):
        guard.check()


def test_warn_mode_records_overspend_without_stopping_the_run() -> None:
    guard = BudgetGuard(Budget(max_usd=1e-9, on_exceed="warn"))
    guard.preflight(calls=100, state_tokens=1000, question_tokens=100)
    guard.record(input_tokens=10_000)
    guard.check()
    assert guard.spent_usd > 0


def test_truncate_mode_reports_how_many_calls_actually_fit() -> None:
    guard = BudgetGuard(Budget(max_usd=estimate_cost(10_000), on_exceed="truncate"))
    affordable = guard.affordable_calls(state_tokens=500, question_tokens=100)
    assert 0 < affordable < 1000


def test_estimates_recalibrate_from_observed_usage() -> None:
    """After one real run the guess is grounded in what the API charged."""
    guard = BudgetGuard(Budget(max_usd=None))
    guard.preflight(calls=10, state_tokens=1000, question_tokens=100)
    for _ in range(10):
        guard.record(input_tokens=2000)
    guard.recalibrate()
    assert guard.calibration > 1.0


def test_no_ceiling_means_no_enforcement() -> None:
    guard = BudgetGuard(Budget(max_usd=None))
    guard.record(input_tokens=10_000_000)
    guard.check()
