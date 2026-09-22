"""Estimator correctness against set functions with known Shapley values.

These tests are the intellectual credibility of the project. They run with no
API key and no network, and they check the estimators against ground truth
rather than against each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev_why.coalitions import (
    CoalitionPlan,
    kernelshap_plan,
    occlusion_plan,
    permutation_plan,
)
from jev_why.estimators import (
    KernelShapEstimator,
    OcclusionEstimator,
    PermutationShapEstimator,
    resolve_estimator,
)
from jev_why.fakes import SyntheticOracle


def test_kernelshap_recovers_known_shapley_values() -> None:
    oracle = SyntheticOracle(n=6, additive={0: 0.3, 4: -0.2},
                             interactions={frozenset({1, 2}): 0.4})
    plan = kernelshap_plan(6, budget_calls=200)
    result = KernelShapEstimator().estimate(plan, oracle.evaluate(plan))
    assert np.allclose(result.phi, oracle.true_shapley(), atol=1e-6)


def test_permutation_and_kernelshap_agree_independently() -> None:
    """Two estimators built on different principles landing on the same answer
    is stronger evidence than either matching its own derivation.

    The tolerance is set by the permutation estimator, which is Monte Carlo and
    converges as 1/sqrt(m). KernelSHAP is exact here; the gap is sampling error
    in the reference, not disagreement about the answer.
    """
    oracle = SyntheticOracle(n=7, additive={i: 0.1 * i for i in range(7)},
                             interactions={frozenset({0, 3, 5}): 0.9})
    truth = oracle.true_shapley()
    kplan = kernelshap_plan(7, budget_calls=400)
    pplan = permutation_plan(7, m_permutations=300)
    kphi = KernelShapEstimator().estimate(kplan, oracle.evaluate(kplan)).phi
    pphi = PermutationShapEstimator().estimate(pplan, oracle.evaluate(pplan)).phi

    assert np.abs(kphi - truth).max() < 1e-6, "kernelshap must be exact on this game"
    assert np.abs(pphi - truth).max() < 0.05, "permutation is Monte Carlo"
    assert np.abs(kphi - pphi).max() < 0.05


def test_shapley_satisfies_efficiency() -> None:
    oracle = SyntheticOracle(n=5, additive={0: 0.5}, interactions={frozenset({2, 3}): 0.3})
    plan = kernelshap_plan(5, budget_calls=120)
    values = oracle.evaluate(plan)
    result = KernelShapEstimator().estimate(plan, values)
    v_full = oracle.value(frozenset(range(5)))
    v_empty = oracle.value(frozenset())
    assert result.phi.sum() == pytest.approx(v_full - v_empty, abs=1e-9)


def test_shapley_gives_dummy_players_nothing() -> None:
    oracle = SyntheticOracle(n=5, additive={0: 0.4, 1: 0.2})
    plan = kernelshap_plan(5, budget_calls=120)
    phi = KernelShapEstimator().estimate(plan, oracle.evaluate(plan)).phi
    assert np.allclose(phi[2:], 0.0, atol=1e-6)


def test_shapley_is_symmetric_for_interchangeable_spans() -> None:
    oracle = SyntheticOracle(n=4, additive={0: 0.25, 1: 0.25})
    plan = kernelshap_plan(4, budget_calls=60)
    phi = KernelShapEstimator().estimate(plan, oracle.evaluate(plan)).phi
    assert phi[0] == pytest.approx(phi[1], abs=1e-6)


def test_leave_one_out_alone_misses_redundant_evidence() -> None:
    """The pathology that justifies the default.

    Two spans each independently justify the answer. Removing either changes
    nothing, so necessity is zero for both and a leave-one-out explanation
    reports that nothing in the document mattered. Sufficiency catches it.
    """
    n = 4
    redundant = {0, 1}

    def value(coalition: frozenset[int]) -> float:
        return 1.0 if coalition & redundant else 0.0

    plan = occlusion_plan(n)
    values = np.array([value(c) for c in plan.coalitions], dtype=np.float64)
    result = OcclusionEstimator().estimate(plan, values)

    assert result.necessity[0] == pytest.approx(0.0)
    assert result.necessity[1] == pytest.approx(0.0)
    assert result.sufficiency[0] == pytest.approx(1.0)
    assert result.phi[0] > 0.4, "averaging necessity and sufficiency must surface the span"


def test_occlusion_overstates_three_way_interactions() -> None:
    """The documented limit of the cheap estimator, asserted rather than hidden.

    A three-way unanimity game gives each member w/3. Occlusion's two-point
    quadrature lands on w/2, and the efficiency gap is what tells the user.
    """
    oracle = SyntheticOracle(n=5, interactions={frozenset({0, 1, 2}): 0.6})
    plan = occlusion_plan(5)
    result = OcclusionEstimator().estimate(plan, oracle.evaluate(plan))

    assert result.phi[0] == pytest.approx(0.3, abs=1e-9)
    assert oracle.true_shapley()[0] == pytest.approx(0.2, abs=1e-9)
    assert result.efficiency_gap > 0.25, "the gap must flag that interactions dominate"


def test_efficiency_gap_is_near_zero_for_additive_states() -> None:
    oracle = SyntheticOracle(n=6, additive={0: 0.3, 1: -0.1, 4: 0.2})
    plan = occlusion_plan(6)
    result = OcclusionEstimator().estimate(plan, oracle.evaluate(plan))
    assert result.efficiency_gap < 1e-9


def test_leave_one_out_alone_still_identifies_the_fit() -> None:
    """n leave-one-out rows plus the efficiency constraint identify n-1 free
    parameters, so this is determined and must not fall back."""
    oracle = SyntheticOracle(n=8, additive={0: 0.5})
    plan = occlusion_plan(8, mode="loo")
    result = KernelShapEstimator().estimate(plan, oracle.evaluate(plan))
    assert result.stderr is not None
    assert result.phi[0] == pytest.approx(0.5, abs=1e-6)


def test_kernelshap_falls_back_rather_than_fitting_underdetermined() -> None:
    """With fewer rows than free parameters, return the cheap estimate rather
    than a confidently wrong fit."""
    oracle = SyntheticOracle(n=8, additive={0: 0.5})
    sparse = CoalitionPlan(
        n_spans=8,
        coalitions=(frozenset(range(8)), frozenset(),
                    frozenset({0}), frozenset({1}), frozenset({0, 1})),
        estimator="shapley",
    )
    result = KernelShapEstimator().estimate(sparse, oracle.evaluate(sparse))
    assert result.stderr is None


def test_kernelshap_tolerates_failed_calls() -> None:
    oracle = SyntheticOracle(n=6, additive={0: 0.4, 2: 0.2})
    plan = kernelshap_plan(6, budget_calls=200)
    values = oracle.evaluate(plan)
    values[5] = np.nan
    values[9] = np.nan
    result = KernelShapEstimator().estimate(plan, values)
    assert result.dropped == 2
    assert np.abs(result.phi - oracle.true_shapley()).max() < 1e-6


def test_resolve_estimator_rejects_unknown_names() -> None:
    assert resolve_estimator("occlusion").name == "occlusion"
    assert resolve_estimator("shapley").name == "shapley"
    with pytest.raises(ValueError, match="unknown estimator"):
        resolve_estimator("magic")
