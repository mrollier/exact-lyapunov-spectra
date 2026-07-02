"""Claim C1: the 16 affine ECAs are exactly the rules with a constant Boolean
Jacobian, with the gradients tabulated in the manuscript (Table 1, Table 2).

The check recomputes the constant-Jacobian property from first principles (a rule
has a constant Jacobian iff every Boolean derivative is independent of the
neighbourhood), rather than trusting any hard-coded list.
"""
import numpy as np
import pytest

from lyapunov.rules import (
    gradient_truth_tables,
    is_affine,
    affine_ecas,
    affine_gradient,
)
from lyapunov.jacobian import eca_jacobian, eca_step

# Table 1 of the manuscript: rule -> gradient (a_-, a_o, a_+), and the complement.
TABLE1 = {
    0: (0, 0, 0), 15: (1, 0, 0), 85: (0, 0, 1), 51: (0, 1, 0),
    60: (1, 1, 0), 102: (0, 1, 1), 90: (1, 0, 1), 150: (1, 1, 1),
}


def _has_constant_jacobian_from_first_principles(rule: int) -> bool:
    """A rule has a constant Jacobian iff each derivative column is constant."""
    gtt = gradient_truth_tables(rule)
    return all(len(set(gtt[i])) == 1 for i in range(3))


def test_constant_jacobian_rules_are_exactly_the_sixteen_affine():
    from_principles = {r for r in range(256) if _has_constant_jacobian_from_first_principles(r)}
    assert from_principles == set(affine_ecas())
    assert len(from_principles) == 16
    # cross-check against the library's own affine predicate
    assert from_principles == {r for r in range(256) if is_affine(r)}


def test_gradients_match_table1_and_complements_share_them():
    for rule, grad in TABLE1.items():
        assert affine_gradient(rule) == grad
        assert affine_gradient(255 - rule) == grad  # a_0 does not enter J


def test_jacobian_is_genuinely_configuration_independent():
    # Recompute the Jacobian by finite differences at random configurations and
    # confirm it never changes -- the defining property of an affine rule.
    rng = np.random.default_rng(0)
    N = 16
    for rule in affine_ecas():
        J_ref = eca_jacobian(rule, N)
        for _ in range(5):
            s = rng.integers(0, 2, size=N)
            J = np.zeros((N, N), dtype=int)
            base = eca_step(rule, s)
            for j in range(N):
                sp = s.copy()
                sp[j] ^= 1
                J[:, j] = (base ^ eca_step(rule, sp))
            assert np.array_equal(J, J_ref)
