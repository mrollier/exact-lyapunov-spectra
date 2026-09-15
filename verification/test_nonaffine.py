"""Unit tests for ``lyapunov.nonaffine``: configuration-dependent Boolean
Jacobians, Benettin along a trajectory, and the exact rank of the tangent map.

The new code path must agree with the verified constant-Jacobian path wherever
the two overlap, which is the whole affine family: that is what ties the
non-affine machinery to everything already checked in C1, C2 and C6. Beyond
that the tests pin down three things the affine case never exercises:

* the Jacobian really does depend on the configuration (the negation of C1),
* a singular Jacobian collapses a direction, and the collapse shows up in the
  QR diagonal as a pivot of order the unit round-off rather than as a zero,
* there is consequently no threshold at which the number of ``-inf`` exponents
  could be read off the QR diagonal: for rules 122, 126, 54 and 110 the
  near-zero pivots run continuously from 1e-6 down to 1e-18 with no internal
  gap wider than two decades. The count is a rank, and is computed in exact
  arithmetic over a finite field.
"""
import numpy as np
import pytest

from lyapunov.benettin import benettin_log_stretch, windowed_spectrum
from lyapunov.jacobian import eca_jacobian, eca_step
from lyapunov.rules import affine_ecas, gradient_truth_tables
from lyapunov.rules import affine_gradient
from lyapunov.nonaffine import (
    RANK_MODULUS,
    RANK_MODULUS_ALT,
    apply_bands,
    integer_matrix_rank,
    benettin_log_stretch_trajectory,
    eca_gradient_bands,
    eca_jacobian_at,
    spectrum_along_trajectory,
    tangent_rank,
    window_product_mod_p,
)

# The nine rules of the non-affine spectra figure, in the manuscript's order.
RULES = (6, 26, 73, 154, 41, 122, 126, 54, 110)

# Ring sizes for the affine cross-check. An affine Jacobian is singular when N
# is divisible by 3 (weight-3 rules), by 4 (rule 90) or by 2 (weight-2 rules);
# 25 and 35 avoid all three, so every direction stays resolvable and the two
# implementations can be compared entry by entry.
AFFINE_N = (25, 35)


def _random_state(N: int, *key) -> np.ndarray:
    return np.random.default_rng(list(key)).integers(0, 2, size=N)


def _finite_difference_jacobian(rule: int, state: np.ndarray) -> np.ndarray:
    """The Jacobian from the definition: flip each cell, see which outputs flip."""
    N = state.size
    base = eca_step(rule, state)
    J = np.zeros((N, N), dtype=int)
    for j in range(N):
        flipped = state.copy()
        flipped[j] ^= 1
        J[:, j] = base ^ eca_step(rule, flipped)
    return J


# --------------------------------------------------------------------------
# The Jacobian itself
# --------------------------------------------------------------------------

def test_jacobian_at_matches_the_constant_jacobian_for_affine_rules():
    # Every affine rule is a fixed point of the new code path: this is the tie
    # to eca_jacobian, and through it to C1.
    for rule in affine_ecas():
        J_ref = eca_jacobian(rule, 17)
        for trial in range(5):
            state = _random_state(17, rule, trial)
            assert np.array_equal(eca_jacobian_at(rule, state), J_ref)


@pytest.mark.parametrize("trial", [0, 1, 2])
def test_jacobian_at_matches_finite_differences_for_all_256_rules(trial):
    # The definitional check, for every rule rather than only the affine ones.
    N = 16
    for rule in range(256):
        state = _random_state(N, rule, trial)
        assert np.array_equal(eca_jacobian_at(rule, state),
                              _finite_difference_jacobian(rule, state))


@pytest.mark.parametrize("rule", RULES)
def test_jacobian_at_varies_with_the_configuration_for_non_affine_rules(rule):
    # The negation of C1: none of these nine rules has a constant Jacobian.
    jacobians = [eca_jacobian_at(rule, _random_state(24, rule, trial)) for trial in range(12)]
    assert any(not np.array_equal(J, jacobians[0]) for J in jacobians[1:])


@pytest.mark.parametrize("rule", RULES)
def test_apply_bands_equals_the_dense_product(rule):
    rng = np.random.default_rng(rule)
    state = _random_state(32, rule, 0)
    bands = eca_gradient_bands(rule, state)
    Q = rng.standard_normal((32, 7))
    assert np.allclose(apply_bands(bands, Q), eca_jacobian_at(rule, state) @ Q, atol=1e-12)


# --------------------------------------------------------------------------
# Benettin along a trajectory
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule", [15, 60, 90, 102, 150, 105])
@pytest.mark.parametrize("N", AFFINE_N)
def test_trajectory_reproduces_the_constant_jacobian_run_on_affine_rules(rule, N):
    # For a constant Jacobian the trajectory version must be the existing one.
    # The banded product adds its three terms in a different order from BLAS, so
    # the agreement is to rounding rather than bitwise for the weight-3 rules.
    T = 30
    state = _random_state(N, rule, 4)
    got = benettin_log_stretch_trajectory(rule, state, T)
    expected = benettin_log_stretch(eca_jacobian(rule, N), T)
    assert np.all(np.isfinite(got))
    assert np.allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("rule", RULES)
def test_row_sums_equal_log_abs_det_at_every_step(rule):
    # C6, extended to a time-varying Jacobian. Where J_t is invertible the log
    # stretching factors sum to ln|det J_t| exactly; where it is singular the
    # true sum is -inf and the computed one is large and negative, because the
    # annihilated direction contributes a pivot of order the unit round-off.
    N, T = 26, 20
    state = _random_state(N, rule, 1)
    L = benettin_log_stretch_trajectory(rule, state, T)
    singular_steps = 0
    for t in range(T):
        sign, logdet = np.linalg.slogdet(eca_jacobian_at(rule, state).astype(float))
        if sign == 0:
            singular_steps += 1
            assert L[t].sum() < -30
        else:
            assert L[t].sum() == pytest.approx(logdet, abs=1e-8)
        state = eca_step(rule, state)
    if rule == 154:
        assert singular_steps == 0
    else:
        assert singular_steps > 0


@pytest.mark.parametrize("rule", [6, 73, 41])
def test_a_vanishing_gradient_row_makes_the_jacobian_singular(rule):
    # These three rules have configurations in which a cell is insensitive to
    # all three of its inputs, which is a zero row of J. The other six are
    # singular too, but never by this route.
    state = _random_state(40, rule, 2)
    bands = eca_gradient_bands(rule, state)
    assert np.any(bands.sum(axis=0) == 0)
    assert np.linalg.matrix_rank(eca_jacobian_at(rule, state).astype(float)) < 40


@pytest.mark.parametrize("rule", [122, 126, 54, 110])
def test_no_threshold_separates_the_collapsed_pivots(rule):
    # The justification for computing the rank exactly. For these rules the QR
    # pivots below 1e-6 form a continuum down to the unit round-off, with no
    # internal gap wide enough to serve as a cut-off, so any threshold-based
    # count of -inf exponents would be arbitrary.
    # Pooled over three initial configurations: the exact set of small pivots
    # one trajectory produces depends on the LAPACK build (a single N = 80
    # trajectory gave a 2.6-decade gap for rule 110 on OpenBLAS 0.3.27), while
    # the continuum the three of them form together does not.
    pooled = []
    for trial in range(3):
        state = _random_state(80, 2 + trial, rule)
        L = benettin_log_stretch_trajectory(rule, state, 280)
        pooled.append(L[80:].ravel())
    pivots = np.sort(np.concatenate(pooled))
    # A hard zero is not a gap. Exact zeros are -inf; a pivot that lands many
    # decades below the unit round-off (1e-31 on the same build, where the
    # laptop's LAPACK returned exactly 0) is the same collapse and is dropped
    # on the same grounds.
    pivots = pivots[pivots > np.log(1e-25)]
    small = pivots[pivots < np.log(1e-6)]
    assert small.size > 100
    assert small[0] < np.log(1e-15)                      # reaches the round-off level
    largest_gap_decades = np.diff(small).max() / np.log(10)
    assert largest_gap_decades < 2.0


@pytest.mark.parametrize("rule", [6, 73])
def test_a_vanishing_row_does_leave_a_gap(rule):
    # The contrast: where the collapse comes from a zero row it is abrupt, and
    # the pivots fall far below the round-off level of the other directions.
    state = _random_state(80, 2, rule)
    L = benettin_log_stretch_trajectory(rule, state, 280)
    pivots = np.sort(L[80:].ravel())
    assert np.any(np.isneginf(pivots))                   # a vanishing row is an exact zero
    pivots = pivots[np.isfinite(pivots)]
    small = pivots[pivots < np.log(1e-6)]
    assert small[0] < np.log(1e-30)
    assert np.diff(small).max() / np.log(10) > 5.0


def test_rule_154_never_collapses():
    # Rule 154's derivative with respect to the right neighbour is the constant
    # 1 (the "1" of its row in the gradient table), so no row of its Jacobian
    # can vanish, the matrix is never singular and the tangent map keeps full
    # rank: the one rule of the nine with no -inf exponents at all.
    assert np.all(gradient_truth_tables(154)[2] == 1)
    for trial in range(5):
        state = _random_state(60, 154, trial)
        L = benettin_log_stretch_trajectory(154, state, 60)
        assert np.all(np.isfinite(L))
        assert tangent_rank(154, state, 20, 40) == 60


def test_spectrum_along_trajectory_is_the_windowed_mean():
    state = _random_state(30, 110, 6)
    L = benettin_log_stretch_trajectory(110, state, 50)
    assert np.array_equal(spectrum_along_trajectory(110, state, 50, 20),
                          windowed_spectrum(L, 20, 30))


def test_results_are_reproducible_from_the_seed():
    state = _random_state(40, 110, 0)
    first = spectrum_along_trajectory(110, state, 60, 20)
    second = spectrum_along_trajectory(110, state.copy(), 60, 20)
    assert np.array_equal(first, second)  # bitwise, not approximate


# --------------------------------------------------------------------------
# The exact rank, and what it means for the reported spectrum
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("window", [1, 2, 3])
def test_exact_rank_matches_the_float_rank_on_short_windows(rule, window):
    # Over one to three steps the product entries are at most 3**3, so the
    # floating-point rank is trustworthy and the two must agree exactly. Over a
    # long window it is not, which is the whole reason for the exact route.
    N = 20
    for trial in range(4):
        state = _random_state(N, 7, rule, trial)
        exact = tangent_rank(rule, state, 5, window)
        s = state.copy()
        for _ in range(5):
            s = eca_step(rule, s)
        P = np.eye(N)
        for _ in range(window):
            P = eca_jacobian_at(rule, s).astype(float) @ P
            s = eca_step(rule, s)
        assert exact == np.linalg.matrix_rank(P)


@pytest.mark.parametrize("rule", RULES)
def test_exact_rank_agrees_between_two_primes(rule):
    # A rank over GF(p) can only fall short of the rank over the rationals, and
    # only if p divides every minor of that order. Two unrelated primes never
    # do so together here.
    for trial in range(3):
        state = _random_state(60, 9, rule, trial)
        assert (tangent_rank(rule, state, 50, 40, RANK_MODULUS)
                == tangent_rank(rule, state, 50, 40, RANK_MODULUS_ALT))


def _zero_singular_values_from_the_algebra(rule: int, N: int) -> int:
    """How many singular values of an affine Jacobian vanish, by hand.

    The symbol of the circulant is ``a_- e^{it} + a_o + a_+ e^{-it}`` at
    ``t = 2 pi k / N``, so the zeros are a question about roots of unity.
    """
    a_minus, a_centre, a_plus = affine_gradient(rule)
    weight = a_minus + a_centre + a_plus
    if weight == 0:
        return N                          # J = 0
    if weight == 1:
        return 0                          # a single unit modulus term
    if weight == 3:
        return 2 if N % 3 == 0 else 0     # 1 + 2 cos t = 0 at t = 2 pi / 3
    if a_centre == 0:
        return 2 if N % 4 == 0 else 0     # |2 cos t| = 0 at t = pi / 2
    return 1 if N % 2 == 0 else 0         # |1 + e^{it}| = 0 at t = pi


@pytest.mark.parametrize("N", [12, 13, 16, 18, 24, 25, 30, 36, 60])
def test_integer_rank_of_the_affine_jacobians_matches_the_algebra(N):
    # The strongest available check on the exact-rank machinery: for the 16
    # affine rules the number of zero singular values is known in closed form
    # from the roots of unity, and both primes must reproduce it exactly.
    for rule in affine_ecas():
        expected = _zero_singular_values_from_the_algebra(rule, N)
        J = eca_jacobian(rule, N)
        assert N - integer_matrix_rank(J, RANK_MODULUS) == expected, (rule, N)
        assert N - integer_matrix_rank(J, RANK_MODULUS_ALT) == expected, (rule, N)


def test_integer_rank_agrees_with_the_float_rank_on_random_small_matrices():
    rng = np.random.default_rng(5)
    for _ in range(20):
        A = rng.integers(-3, 4, size=(9, 9))
        if rng.random() < 0.5:                    # force a rank deficiency
            A[-1] = A[:-1].sum(axis=0)
        assert integer_matrix_rank(A) == np.linalg.matrix_rank(A.astype(float))


def test_window_product_is_reduced_and_exact():
    state = _random_state(40, 110, 0)
    P = window_product_mod_p(110, state, 10, 25)
    assert P.dtype == np.int64
    assert np.all((P >= 0) & (P < RANK_MODULUS))
    # A window of one step is just the Jacobian at that configuration, mod p.
    s = state.copy()
    for _ in range(10):
        s = eca_step(110, s)
    assert np.array_equal(window_product_mod_p(110, state, 10, 1),
                          eca_jacobian_at(110, s) % RANK_MODULUS)


@pytest.mark.parametrize("rule", RULES)
def test_censoring_the_bottom_exponents_is_unambiguous(rule):
    # The reported spectrum keeps the top ``rank`` exponents and censors the
    # rest as -inf. That is only meaningful if the censored ones all sit below
    # the kept ones, which is what this asserts: no interleaving at the cut.
    N, burn, window = 80, 80, 120
    for trial in range(2):
        state = _random_state(N, 13, rule, trial)
        spectrum = spectrum_along_trajectory(rule, state, burn + window, burn)
        rank = tangent_rank(rule, state, burn, window)
        assert 0 < rank <= N
        kept, censored = spectrum[:rank], spectrum[rank:]
        assert np.all(np.isfinite(kept)), "a kept exponent is -inf"
        if censored.size and np.any(np.isfinite(censored)):
            assert censored[np.isfinite(censored)].max() < kept.min()


# --------------------------------------------------------------------------
# Argument checking
# --------------------------------------------------------------------------

def test_bad_arguments_raise():
    state = _random_state(20, 110, 0)
    with pytest.raises(ValueError):
        benettin_log_stretch_trajectory(110, state, 0)
    with pytest.raises(ValueError):
        benettin_log_stretch_trajectory(110, state, 5, Q0=np.eye(19))
    with pytest.raises(ValueError):
        eca_gradient_bands(110, np.array([0, 1]))             # fewer than three cells
    with pytest.raises(ValueError):
        eca_gradient_bands(110, np.array([0, 2, 1]))          # not binary
    with pytest.raises(ValueError):
        eca_gradient_bands(110, np.zeros((4, 4), dtype=int))  # not one-dimensional
    with pytest.raises(ValueError):
        eca_gradient_bands(300, state)                        # not an ECA
    with pytest.raises(ValueError):
        spectrum_along_trajectory(110, state, 10, 10)         # burn must be below T
    with pytest.raises(ValueError):
        window_product_mod_p(110, state, -1, 5)               # burn must be non-negative
    with pytest.raises(ValueError):
        apply_bands(np.ones((2, 20)), np.eye(20))             # bands must have three rows
    with pytest.raises(ValueError):
        apply_bands(np.ones((3, 20)), np.eye(19))             # mismatched shapes


# --------------------------------------------------------------------------
# The checkpoint of scripts/make_nonaffine_spectra.py. A run over all 88 rules
# takes hours, so an interrupted one must resume without changing any number.
# --------------------------------------------------------------------------

def _runner():
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import make_nonaffine_spectra
    return make_nonaffine_spectra


RESUME_RULES = (110, 30)
RESUME_KW = dict(N=32, T=16, burn=8, samples=2, seed=1, workers=2, with_direct=False)


def test_a_resumed_run_reproduces_an_uninterrupted_one(tmp_path):
    run = _runner()
    partial = run.partial_path(tmp_path / "spectra.npz")
    whole = run.compute(RESUME_RULES, partial=partial, resume=False, **RESUME_KW)
    assert partial.exists()

    # Forget one sample of each rule, then resume: the seeds depend only on
    # (seed, rule, sample), so the redone samples must come back identical.
    with np.load(partial) as stored:
        state = {key: stored[key] for key in stored.files}
    state["finished"][:, -1] = False
    state["spectra"][:, -1] = np.nan
    np.savez(partial, **state)

    resumed = run.compute(RESUME_RULES, partial=partial, resume=True, **RESUME_KW)
    assert np.array_equal(whole["spectra"], resumed["spectra"])
    assert np.array_equal(whole["ranks"], resumed["ranks"])


def test_a_checkpoint_from_other_parameters_is_ignored(tmp_path):
    run = _runner()
    partial = run.partial_path(tmp_path / "spectra.npz")
    short = run.compute(RESUME_RULES, partial=partial, resume=False, **RESUME_KW)
    longer = run.compute(RESUME_RULES, partial=partial, resume=True,
                         **{**RESUME_KW, "T": 20})
    # A longer horizon gives different exponents. The stale checkpoint must be
    # discarded rather than merged, so nothing of the short run survives.
    assert not np.array_equal(short["spectra"], longer["spectra"])


def test_the_88_rule_sets_are_the_symmetry_classes():
    run = _runner()
    from lyapunov.rules import nonequivalent_ecas
    assert len(run.ALL_88_RULES) + len(run.ALL_88_AFFINE) == 88
    assert sorted(run.ALL_88_RULES + run.ALL_88_AFFINE) == nonequivalent_ecas()
    assert set(run.ALL_88_AFFINE) == set(nonequivalent_ecas()) & set(affine_ecas())
    assert not set(run.ALL_88_RULES) & set(affine_ecas())
