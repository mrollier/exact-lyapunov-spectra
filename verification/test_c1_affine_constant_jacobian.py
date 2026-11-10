"""Claim C1: the 16 affine ECAs are exactly the rules with a constant Boolean
Jacobian, with the gradients tabulated in the manuscript (Tab. 3), and such a
rule is reversible exactly when det J is odd (Subsec. 3.1).

The check recomputes the constant-Jacobian property from first principles (a rule
has a constant Jacobian iff every Boolean derivative is independent of the
neighbourhood), rather than trusting any hard-coded list.
"""
import numpy as np

from lyapunov.rules import (
    truth_table,
    gradient_truth_tables,
    is_affine,
    affine_ecas,
    affine_gradient,
)
from lyapunov.jacobian import eca_jacobian, eca_step

# Tab. 3 of the manuscript: rule -> gradient (a_-, a_o, a_+); the complement
# 255 - rule shares it. Same rows and order as scripts/make_tables.py.
TAB3 = {
    0: (0, 0, 0), 170: (0, 0, 1), 204: (0, 1, 0), 240: (1, 0, 0),
    60: (1, 1, 0), 90: (1, 0, 1), 102: (0, 1, 1), 150: (1, 1, 1),
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


def test_gradients_match_tab3_and_complements_share_them():
    for rule, grad in TAB3.items():
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


def test_affine_rule_is_reversible_iff_det_j_is_odd():
    # Subsec. 3.1: Phi is a bijection iff J is invertible mod 2, i.e. det J odd.
    # Checked by enumerating every configuration on rings of 3 to 10 cells.
    for N in range(3, 11):
        codes = np.arange(2**N)
        s = (codes[:, None] >> np.arange(N)) & 1
        code = 4 * np.roll(s, 1, axis=1) + 2 * s + np.roll(s, -1, axis=1)
        for rule in affine_ecas():
            image = truth_table(rule)[code] @ (1 << np.arange(N))
            bijective = len(np.unique(image)) == 2**N
            det = round(np.linalg.det(eca_jacobian(rule, N).astype(float)))
            assert bijective == (det % 2 == 1), (rule, N, det)
            if rule == 150 and N % 3:
                # Subsec. 3.1: reversible, yet tangent-space volume triples per step.
                assert bijective and abs(det) == 3
