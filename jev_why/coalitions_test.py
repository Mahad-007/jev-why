from __future__ import annotations

import pytest

from jev_why.coalitions import (
    EMPTY,
    kernel_weight,
    kernelshap_plan,
    occlusion_plan,
    permutation_plan,
)


def test_occlusion_costs_two_n_plus_two() -> None:
    plan = occlusion_plan(10)
    assert plan.n_calls == 22
    assert frozenset(range(10)) in plan.coalitions
    assert EMPTY in plan.coalitions


def test_occlusion_modes_trade_calls_for_coverage() -> None:
    assert occlusion_plan(10, mode="loo").n_calls == 12
    assert occlusion_plan(10, mode="loi").n_calls == 12


def test_shapley_plan_extends_the_occlusion_plan_in_place() -> None:
    """The nesting property. Upgrading a cached occlusion run to Shapley must
    pay only the difference, so the cheap plan has to be a literal prefix."""
    occlusion = occlusion_plan(8)
    shapley = kernelshap_plan(8, budget_calls=120)
    assert shapley.coalitions[: occlusion.n_calls] == occlusion.coalitions


def test_plans_never_repeat_a_coalition() -> None:
    """A duplicate is a call paid for twice and a row that double-counts in the
    regression."""
    for plan in (kernelshap_plan(9, budget_calls=200), permutation_plan(7, m_permutations=40)):
        assert len(set(plan.coalitions)) == plan.n_calls


def test_shapley_budget_is_respected() -> None:
    plan = kernelshap_plan(12, budget_calls=80)
    assert plan.n_calls <= 80


def test_shapley_budget_cannot_fall_below_the_occlusion_floor() -> None:
    """Asking for less than the occlusion plan would drop the leave-one-out and
    leave-one-in strata, which carry the highest kernel weight."""
    plan = kernelshap_plan(6, budget_calls=4)
    assert plan.n_calls >= occlusion_plan(6).n_calls


def test_kernel_weight_peaks_at_the_occlusion_strata() -> None:
    n = 10
    weights = {k: kernel_weight(n, k) for k in range(1, n)}
    assert max(weights, key=lambda k: weights[k]) in (1, n - 1)
    assert weights[1] == pytest.approx(weights[n - 1])


def test_plans_are_deterministic_under_a_seed() -> None:
    assert (
        kernelshap_plan(9, budget_calls=90, seed=7).coalitions
        == kernelshap_plan(9, budget_calls=90, seed=7).coalitions
    )


def test_permutation_plan_records_the_orders_it_walked() -> None:
    plan = permutation_plan(5, m_permutations=3)
    perms = plan.meta["permutations"]
    assert len(perms) == 6, "each draw contributes its own reverse"
    assert all(sorted(p) == list(range(5)) for p in perms)
