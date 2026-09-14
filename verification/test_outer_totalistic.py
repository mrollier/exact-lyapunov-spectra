"""Unit tests for ``lyapunov.outer_totalistic``: the 2-D outer-totalistic rules
on the von Neumann neighbourhood, their symmetry classes, the torus step, and
the configuration-dependent Boolean Jacobian.

The family is small enough to check exhaustively: every one of the 1024 rules
has its Jacobian bands compared with brute-force single-cell flips on random
configurations, and the class count (528) is derived rather than assumed. The
two parity rules tie the new code to the closed form of ``lyapunov.spectra``:
their Jacobian is constant and equals the torus adjacency of
``lyapunov.jacobian.build_torus_parity_jacobian``.

The Moore neighbourhood (the "Life-like" family, 2**18 rules in 131 328 classes)
enters through the ``neighbourhood`` argument. It is too large to check
exhaustively, so its Jacobian is compared with brute force on a seeded random
subset of rules plus Life and the two Moore parity rules, and the class count
is derived from a vectorised conjugation checked against the per-rule loop.
"""
import numpy as np
import pytest

from lyapunov.jacobian import build_torus_parity_jacobian
from lyapunov.spectra import MOORE_2D, VON_NEUMANN_2D
from lyapunov.outer_totalistic import (
    MOORE,
    NEIGHBOUR_OFFSETS,
    PARITY_MOORE_EXCLUSIVE,
    PARITY_MOORE_INCLUSIVE,
    PARITY_VN_EXCLUSIVE,
    PARITY_VN_INCLUSIVE,
    VON_NEUMANN,
    apply_bands_2d,
    bs_notation,
    conjugate_rule,
    nonequivalent_outer_totalistic,
    ot_from_bs,
    ot_gradient_bands,
    ot_jacobian_at,
    ot_step,
    ot_table,
)


def random_state(L: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=(L, L))


def naive_step(rule: int, state: np.ndarray) -> np.ndarray:
    """The definition, one cell at a time, with explicit periodic indices."""
    table = ot_table(rule)
    L = state.shape[0]
    out = np.empty_like(state)
    for i in range(L):
        for j in range(L):
            n = (state[(i + 1) % L, j] + state[(i - 1) % L, j]
                 + state[i, (j + 1) % L] + state[i, (j - 1) % L])
            out[i, j] = table[state[i, j], n]
    return out


# --------------------------------------------------------------------------
# Encoding and notation
# --------------------------------------------------------------------------

def test_table_bit_layout():
    # Bit 5c + n is f(c, n). Rule 1 << 7 is "a live cell with two live
    # neighbours stays alive, nothing else ever happens".
    table = ot_table(1 << 7)
    assert table.shape == (2, 5)
    assert table[1, 2] == 1 and table.sum() == 1


@pytest.mark.parametrize("rule", [0, 1, 1023, PARITY_VN_INCLUSIVE, PARITY_VN_EXCLUSIVE, 682])
def test_bs_notation_round_trip(rule):
    assert ot_from_bs(bs_notation(rule)) == rule


def test_bs_notation_examples():
    assert bs_notation(0) == "B/S"
    assert bs_notation(1023) == "B01234/S01234"
    assert bs_notation(PARITY_VN_INCLUSIVE) == "B13/S024"
    assert bs_notation(PARITY_VN_EXCLUSIVE) == "B13/S13"


def test_parity_rules_are_parity():
    inc, exc = ot_table(PARITY_VN_INCLUSIVE), ot_table(PARITY_VN_EXCLUSIVE)
    for c in (0, 1):
        for n in range(5):
            assert inc[c, n] == (c + n) % 2
            assert exc[c, n] == n % 2


@pytest.mark.parametrize("rule", [5, 77, 1000])
def test_rule_range_checked(rule):
    with pytest.raises(ValueError):
        ot_table(-1)
    with pytest.raises(ValueError):
        ot_table(1024)
    ot_table(rule)   # in range: fine


# --------------------------------------------------------------------------
# Symmetry classes
# --------------------------------------------------------------------------

def test_conjugation_is_an_involution_with_32_fixed_points():
    fixed = 0
    for rule in range(1024):
        assert conjugate_rule(conjugate_rule(rule)) == rule
        fixed += conjugate_rule(rule) == rule
    # f(c, n) = 1 - f(1 - c, 4 - n) fixes the 10 bits in 5 pairs: 2**5 choices.
    assert fixed == 32


def test_conjugation_commutes_with_black_white_swap_of_configurations():
    # Conjugate rule on the complemented configuration = complement of the
    # rule on the configuration. This is what makes the two rules equivalent.
    state = random_state(7, 1)
    for rule in (3, 682, 1000, 511):
        lhs = ot_step(conjugate_rule(rule), 1 - state)
        rhs = 1 - ot_step(rule, state)
        assert np.array_equal(lhs, rhs)


def test_528_nonequivalent_rules():
    reps = nonequivalent_outer_totalistic()
    assert len(reps) == 528 == (1024 + 32) // 2
    assert reps == sorted(reps)
    assert all(r <= conjugate_rule(r) for r in reps)      # minimal representative
    assert set(reps) | {conjugate_rule(r) for r in reps} == set(range(1024))
    assert PARITY_VN_INCLUSIVE in reps and PARITY_VN_EXCLUSIVE in reps


# --------------------------------------------------------------------------
# Evolution
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule", [0, 1023, PARITY_VN_INCLUSIVE, 682, 301, 777])
def test_step_matches_naive_definition(rule):
    state = random_state(9, rule)
    assert np.array_equal(ot_step(rule, state), naive_step(rule, state))


def test_step_rejects_bad_input():
    with pytest.raises(ValueError):
        ot_step(1, np.zeros(9, dtype=int))          # not 2-D
    with pytest.raises(ValueError):
        ot_step(1, np.full((4, 4), 2))              # not binary


def test_inclusive_parity_step_is_linear_over_gf2():
    a, b = random_state(8, 3), random_state(8, 4)
    lhs = ot_step(PARITY_VN_INCLUSIVE, a ^ b)
    rhs = ot_step(PARITY_VN_INCLUSIVE, a) ^ ot_step(PARITY_VN_INCLUSIVE, b)
    assert np.array_equal(lhs, rhs)


# --------------------------------------------------------------------------
# Boolean Jacobian
# --------------------------------------------------------------------------

def brute_force_bands(rule: int, state: np.ndarray) -> np.ndarray:
    """Derivative of cell i's output w.r.t. each of its five inputs, by flipping."""
    L = state.shape[0]
    base = ot_step(rule, state)
    bands = np.zeros((5, L, L), dtype=int)
    offsets = ((0, 0),) + tuple(NEIGHBOUR_OFFSETS)
    for k, (dx, dy) in enumerate(offsets):
        for i in range(L):
            for j in range(L):
                flipped = state.copy()
                flipped[(i + dx) % L, (j + dy) % L] ^= 1
                bands[k, i, j] = base[i, j] ^ ot_step(rule, flipped)[i, j]
    return bands


@pytest.mark.parametrize("rule", range(0, 1024, 37))   # 28 rules spread over the family
def test_gradient_bands_match_brute_force_flips(rule):
    state = random_state(6, 100 + rule)
    assert np.array_equal(ot_gradient_bands(rule, state), brute_force_bands(rule, state))


def test_every_rule_gradient_bands_on_one_configuration():
    state = random_state(5, 2024)
    for rule in range(1024):
        assert np.array_equal(ot_gradient_bands(rule, state), brute_force_bands(rule, state))


def test_parity_jacobian_is_the_torus_adjacency():
    L = 5
    state = random_state(L, 11)
    J = ot_jacobian_at(PARITY_VN_INCLUSIVE, state)
    assert np.array_equal(J, build_torus_parity_jacobian(VON_NEUMANN_2D, L))
    J = ot_jacobian_at(PARITY_VN_EXCLUSIVE, state)
    assert np.array_equal(J, build_torus_parity_jacobian(VON_NEUMANN_2D[1:], L))


def test_apply_bands_2d_equals_dense_product():
    L = 7
    state = random_state(L, 5)
    v = np.random.default_rng(6).standard_normal((L, L))
    for rule in (682, 301, 1023, 3):
        bands = ot_gradient_bands(rule, state)
        dense = ot_jacobian_at(rule, state) @ v.ravel()
        assert np.allclose(apply_bands_2d(bands, v).ravel(), dense)


def test_apply_bands_2d_rejects_shape_mismatch():
    bands = np.zeros((5, 4, 4))
    with pytest.raises(ValueError):
        apply_bands_2d(bands, np.zeros((4, 5)))
    with pytest.raises(ValueError):
        apply_bands_2d(np.zeros((4, 4, 4)), np.zeros((4, 4)))


# --------------------------------------------------------------------------
# The Moore neighbourhood
# --------------------------------------------------------------------------

LIFE = ot_from_bs("B3/S23", MOORE)
MOORE_RULES = [int(r) for r in np.random.default_rng(2024).integers(0, 1 << 18, 200)]


def naive_step_moore(rule: int, state: np.ndarray) -> np.ndarray:
    table = ot_table(rule, MOORE)
    L = state.shape[0]
    out = np.empty_like(state)
    for i in range(L):
        for j in range(L):
            n = sum(state[(i + dx) % L, (j + dy) % L]
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))
            out[i, j] = table[state[i, j], n]
    return out


def brute_force_bands_moore(rule: int, state: np.ndarray) -> np.ndarray:
    L = state.shape[0]
    base = ot_step(rule, state, MOORE)
    bands = np.zeros((9, L, L), dtype=int)
    for k, (dx, dy) in enumerate(((0, 0),) + tuple(MOORE.offsets)):
        for i in range(L):
            for j in range(L):
                flipped = state.copy()
                flipped[(i + dx) % L, (j + dy) % L] ^= 1
                bands[k, i, j] = base[i, j] ^ ot_step(rule, flipped, MOORE)[i, j]
    return bands


def test_neighbourhood_constants():
    assert VON_NEUMANN.offsets == NEIGHBOUR_OFFSETS and VON_NEUMANN.k == 4
    assert VON_NEUMANN.counts == 5 and VON_NEUMANN.n_rules == 1024
    assert MOORE.k == 8 and MOORE.counts == 9 and MOORE.n_rules == 1 << 18
    assert set(MOORE.offsets) == set(MOORE_2D) - {(0, 0)}
    assert len(set(MOORE.offsets)) == 8


def test_moore_table_and_notation():
    # Bit 9c + n is f(c, n); Life is B3/S23.
    assert LIFE == (1 << 3) | (1 << 11) | (1 << 12)
    assert ot_table(LIFE, MOORE).shape == (2, 9)
    assert bs_notation(LIFE, MOORE) == "B3/S23"
    assert bs_notation(0, MOORE) == "B/S"
    assert bs_notation((1 << 18) - 1, MOORE) == "B012345678/S012345678"
    for rule in (0, 1, LIFE, PARITY_MOORE_INCLUSIVE, (1 << 18) - 1) + tuple(MOORE_RULES[:5]):
        assert ot_from_bs(bs_notation(rule, MOORE), MOORE) == rule
    with pytest.raises(ValueError):
        ot_from_bs("B9/S", MOORE)
    with pytest.raises(ValueError):
        ot_table(1 << 18, MOORE)
    with pytest.raises(ValueError):
        ot_table(1024, VON_NEUMANN)


def test_moore_parity_rules_are_parity():
    inc, exc = ot_table(PARITY_MOORE_INCLUSIVE, MOORE), ot_table(PARITY_MOORE_EXCLUSIVE, MOORE)
    for c in (0, 1):
        for n in range(9):
            assert inc[c, n] == (c + n) % 2
            assert exc[c, n] == n % 2


def test_moore_conjugation_and_class_count():
    # f'(c, n) = 1 - f(1 - c, 8 - n) pairs the 18 bits: 2**9 fixed points.
    for rule in MOORE_RULES[:50] + [0, LIFE, PARITY_MOORE_INCLUSIVE]:
        assert conjugate_rule(conjugate_rule(rule, MOORE), MOORE) == rule
    state = random_state(7, 8)
    for rule in (LIFE, 12345, 200000):
        assert np.array_equal(ot_step(conjugate_rule(rule, MOORE), 1 - state, MOORE),
                              1 - ot_step(rule, state, MOORE))
    reps = nonequivalent_outer_totalistic(MOORE)
    assert len(reps) == 131328 == ((1 << 18) + 512) // 2
    assert reps == sorted(reps) and len(set(reps)) == len(reps)
    for rule in reps[:1000] + reps[-1000:]:
        assert rule <= conjugate_rule(rule, MOORE)
    assert LIFE in reps


def test_vectorised_representatives_agree_with_the_loop():
    reps = nonequivalent_outer_totalistic(VON_NEUMANN)
    assert reps == sorted(r for r in range(1024) if r <= conjugate_rule(r))
    assert nonequivalent_outer_totalistic() == reps


@pytest.mark.parametrize("rule", [LIFE, 0, PARITY_MOORE_INCLUSIVE] + MOORE_RULES[:3])
def test_moore_step_matches_naive_definition(rule):
    state = random_state(9, rule % 1000)
    assert np.array_equal(ot_step(rule, state, MOORE), naive_step_moore(rule, state))


def test_life_blinker():
    L = 7
    state = np.zeros((L, L), dtype=int)
    state[3, 2:5] = 1
    after = ot_step(LIFE, state, MOORE)
    expected = np.zeros((L, L), dtype=int)
    expected[2:5, 3] = 1
    assert np.array_equal(after, expected)
    assert np.array_equal(ot_step(LIFE, after, MOORE), state)


@pytest.mark.parametrize("rule", [LIFE, PARITY_MOORE_INCLUSIVE, PARITY_MOORE_EXCLUSIVE])
def test_moore_gradient_bands_match_brute_force_flips(rule):
    state = random_state(7, 300 + rule % 100)
    assert np.array_equal(ot_gradient_bands(rule, state, MOORE), brute_force_bands_moore(rule, state))


def test_random_moore_rules_gradient_bands_on_one_configuration():
    state = random_state(5, 2025)
    for rule in MOORE_RULES:
        assert np.array_equal(ot_gradient_bands(rule, state, MOORE), brute_force_bands_moore(rule, state))


def test_moore_parity_jacobian_is_the_torus_adjacency():
    L = 5
    state = random_state(L, 12)
    J = ot_jacobian_at(PARITY_MOORE_INCLUSIVE, state, MOORE)
    assert np.array_equal(J, build_torus_parity_jacobian(MOORE_2D, L))
    J = ot_jacobian_at(PARITY_MOORE_EXCLUSIVE, state, MOORE)
    assert np.array_equal(J, build_torus_parity_jacobian([o for o in MOORE_2D if o != (0, 0)], L))


def test_apply_bands_2d_infers_the_moore_neighbourhood():
    L = 7
    state = random_state(L, 13)
    v = np.random.default_rng(14).standard_normal((L, L))
    for rule in (LIFE, PARITY_MOORE_INCLUSIVE) + tuple(MOORE_RULES[:3]):
        bands = ot_gradient_bands(rule, state, MOORE)
        dense = ot_jacobian_at(rule, state, MOORE) @ v.ravel()
        assert np.allclose(apply_bands_2d(bands, v).ravel(), dense)
    with pytest.raises(ValueError):
        apply_bands_2d(np.zeros((6, L, L)), v)
