"""Unit tests for ``lyapunov.damage``: Boolean damage from a single flipped
cell, its normalisation by the light cone, and the single-vector estimate of
the maximal Lyapunov exponent.

The damage facts are exact and elementary: the identity keeps one damaged
cell for ever, the shift moves it at one cell per step, the null rule heals it
at once, and for the affine rules the damage pattern *is* the mod-2 defect
pattern of the paper, so rule 90 damages ``2**popcount(t)`` cells (Sierpinski)
and the 2-D parity rule damages exactly the odd entries of ``(A + I)**t e_j``,
computed with the overflow-safe ``gf2_matrix_power``.

The single-vector estimator is tied to the constant-Jacobian closed form: on
an affine rule the per-step log stretches of a vector ``v`` under a normal
circulant ``J`` are known exactly from the Fourier coefficients of ``v``, so
the estimator must reproduce ``ln(|J**T v| / |J**burn v|) / (T - burn)`` to
rounding, and be within a stated distance of the exact ``ln 3`` / ``ln 2``.
"""
import numpy as np
import pytest

from lyapunov.gf2 import gf2_matrix_power
from lyapunov.jacobian import build_torus_parity_jacobian, eca_step
from lyapunov.nonaffine import apply_bands, eca_gradient_bands, spectrum_along_trajectory
from lyapunov.outer_totalistic import MOORE, PARITY_MOORE_INCLUSIVE, PARITY_VN_INCLUSIVE, ot_from_bs, ot_step
from lyapunov.spectra import MOORE_2D, VON_NEUMANN_2D, eca_eigenvalues, eca_mle
from lyapunov.damage import (
    damage_series,
    damage_summary,
    eca_damage_and_mle,
    maximal_cone,
    mle_single_vector,
    observed_cone,
    ot_damage_and_mle,
)


def ring(N: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=N)


def eca_damage(rule, N, T, seed):
    return damage_series(lambda s: eca_step(rule, s), ring(N, seed), N // 2, T)


# --------------------------------------------------------------------------
# Cones
# --------------------------------------------------------------------------

def test_cone_sizes():
    t = np.arange(4)
    assert np.array_equal(maximal_cone(t, 1), [1, 3, 5, 7])
    assert np.array_equal(maximal_cone(t, 2), [1, 5, 13, 25])   # L1 balls (diamonds)
    assert np.array_equal(observed_cone(t, 2), maximal_cone(t, 2))
    assert np.array_equal(maximal_cone(t, 2, moore=True), [1, 9, 25, 49])   # L-inf balls (squares)
    assert np.array_equal(observed_cone(t, 2, moore=True), maximal_cone(t, 2, moore=True))
    with pytest.raises(ValueError):
        maximal_cone(t, 3)
    with pytest.raises(ValueError):
        maximal_cone(t, 1, moore=True)


# --------------------------------------------------------------------------
# Damage series, 1-D
# --------------------------------------------------------------------------

def test_identity_keeps_one_damaged_cell():
    h, r = eca_damage(204, 41, 19, 1)
    assert np.all(h == 1) and np.all(r == 0)


def test_shift_moves_the_damage_at_speed_one():
    h, r = eca_damage(170, 41, 19, 2)
    assert np.all(h == 1)
    assert np.array_equal(r, np.arange(20))


def test_null_rule_heals_at_once():
    h, r = eca_damage(0, 41, 19, 3)
    assert h[0] == 1 and np.all(h[1:] == 0) and np.all(r[1:] == 0)


def test_rule_90_damage_is_sierpinski():
    T = 30
    h, r = eca_damage(90, 2 * T + 3, T, 4)
    expected = [2 ** bin(t).count("1") for t in range(T + 1)]
    assert np.array_equal(h, expected)
    # the outermost damaged cells sit exactly on the light cone
    assert np.array_equal(r, np.arange(T + 1))


def test_damage_stays_inside_the_light_cone():
    T = 25
    for rule in (30, 110, 150, 54, 90, 22):
        h, r = eca_damage(rule, 2 * T + 3, T, rule)
        assert np.all(h <= maximal_cone(np.arange(T + 1), 1))
        assert np.all(r <= np.arange(T + 1))


def test_damage_of_affine_rule_is_independent_of_the_configuration():
    # Affine rule: the difference pattern is the mod-2 defect pattern, which
    # does not depend on the configuration it is superposed on.
    h1, _ = eca_damage(150, 43, 20, 5)
    h2, _ = eca_damage(150, 43, 20, 6)
    assert np.array_equal(h1, h2)


def test_wrap_guard():
    with pytest.raises(ValueError, match="wrap"):
        eca_damage(30, 40, 19, 1)    # 40 < 2*19 + 3
    eca_damage(30, 41, 19, 1)        # 41 = 2*19 + 3 is allowed


def test_damage_series_rejects_bad_flip_index():
    with pytest.raises(ValueError):
        damage_series(lambda s: eca_step(30, s), ring(41, 1), (3, 4), 19)


# --------------------------------------------------------------------------
# Damage series, 2-D
# --------------------------------------------------------------------------

def test_2d_parity_damage_is_the_gf2_defect_pattern():
    L, T = 11, 4
    state = np.random.default_rng(7).integers(0, 2, size=(L, L))
    h, r = damage_series(lambda s: ot_step(PARITY_VN_INCLUSIVE, s), state, (L // 2, L // 2), T)
    A = build_torus_parity_jacobian(VON_NEUMANN_2D, L)      # A + I on the torus
    e = np.zeros(L * L, dtype=int)
    e[(L // 2) * L + L // 2] = 1
    for t in range(T + 1):
        pattern = (gf2_matrix_power(A, t) @ e) % 2
        assert h[t] == pattern.sum()
    # Sierpinski-like, not a full diamond: h = 1, 5, 5, 17, 5. The four tips of
    # the diamond are reached by exactly one path, so they are always damaged
    # and the front moves at speed one.
    assert np.array_equal(h, [1, 5, 5, 17, 5])
    assert np.array_equal(r, np.arange(T + 1))


def test_2d_damage_stays_inside_the_diamond():
    L, T = 23, 10
    state = np.random.default_rng(8).integers(0, 2, size=(L, L))
    for rule in (301, 777, 1000, 511):
        h, r = damage_series(lambda s: ot_step(rule, s), state, (L // 2, L // 2), T)
        assert np.all(h <= maximal_cone(np.arange(T + 1), 2))
        assert np.all(r <= np.arange(T + 1))


def test_2d_moore_parity_damage_is_the_gf2_defect_pattern():
    L, T = 11, 4
    state = np.random.default_rng(9).integers(0, 2, size=(L, L))
    h, r = damage_series(lambda s: ot_step(PARITY_MOORE_INCLUSIVE, s, MOORE), state,
                         (L // 2, L // 2), T, moore=True)
    A = build_torus_parity_jacobian(MOORE_2D, L)             # 3 x 3 block, self included
    e = np.zeros(L * L, dtype=int)
    e[(L // 2) * L + L // 2] = 1
    for t in range(T + 1):
        assert h[t] == ((gf2_matrix_power(A, t) @ e) % 2).sum()
    # The 3 x 3 kernel is the tensor square of rule 150's kernel, so over GF(2)
    # the damage is the tensor square of rule 150's pattern: its counts
    # 1, 3, 3, 5, 3 squared give h = 1, 9, 9, 25, 9. The radius, measured in
    # the Chebyshev metric, moves at speed one.
    assert np.array_equal(h, [1, 9, 9, 25, 9])
    assert np.array_equal(r, np.arange(T + 1))


def test_2d_moore_damage_stays_inside_the_square_and_uses_the_chebyshev_radius():
    L, T = 23, 10
    state = np.random.default_rng(10).integers(0, 2, size=(L, L))
    life = ot_from_bs("B3/S23", MOORE)
    for rule in (life, 12345, 200000, (1 << 18) - 1):
        h, r = damage_series(lambda s: ot_step(rule, s, MOORE), state, (L // 2, L // 2), T, moore=True)
        assert np.all(h <= maximal_cone(np.arange(T + 1), 2, moore=True))
        assert np.all(r <= np.arange(T + 1))
    # A rule that copies the corner neighbour moves the damage diagonally: L1
    # radius 2t, Chebyshev radius t.
    diag = lambda s: np.roll(np.roll(s, 1, 0), 1, 1)
    _, r_cheb = damage_series(diag, state, (L // 2, L // 2), T, moore=True)
    _, r_l1 = damage_series(diag, state, (L // 2, L // 2), T)
    assert np.array_equal(r_cheb, np.arange(T + 1))
    assert np.array_equal(r_l1, 2 * np.arange(T + 1))


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def test_summary_of_shift_and_identity():
    h, r = eca_damage(170, 41, 19, 2)
    s = damage_summary(h, r, 1, window=5)
    assert s["v_front"] == pytest.approx(1.0)
    assert s["fill"] == pytest.approx(np.mean([1 / (2 * t + 1) for t in range(15, 20)]))
    assert s["D_norm"] == s["fill"]                         # observed cone = maximal cone
    assert s["h_final"] == 1 and s["r_final"] == 19
    h, r = eca_damage(204, 41, 19, 1)
    s = damage_summary(h, r, 1, window=5)
    assert s["v_front"] == 0 and s["fill"] == 1.0
    assert s["D_norm"] == pytest.approx(np.mean([1 / (2 * t + 1) for t in range(15, 20)]))


def test_summary_uses_the_square_cone_for_moore():
    h = np.array([1, 9, 25, 49, 81])
    r = np.arange(5)
    s = damage_summary(h, r, 2, window=2, moore=True)
    assert s["D_norm"] == pytest.approx(1.0) and s["fill"] == pytest.approx(1.0)
    assert damage_summary(h, r, 2, window=2)["D_norm"] > 1     # the diamond is smaller
    with pytest.raises(ValueError):
        damage_summary(h, r, 1, window=2, moore=True)


def test_summary_of_healed_damage_is_zero():
    h, r = eca_damage(0, 41, 19, 3)
    s = damage_summary(h, r, 1, window=5)
    assert s["v_front"] == 0 and s["D_norm"] == 0 and s["fill"] == 0


def test_summary_window_bounds():
    h, r = eca_damage(30, 41, 19, 3)
    with pytest.raises(ValueError):
        damage_summary(h, r, 1, window=0)
    with pytest.raises(ValueError):
        damage_summary(h, r, 1, window=20)     # the window excludes t = 0


# --------------------------------------------------------------------------
# Single-vector maximal exponent
# --------------------------------------------------------------------------

def eca_mle_1v(rule, state, T, burn, seed):
    return mle_single_vector(
        lambda s: eca_gradient_bands(rule, s),
        lambda b, v: apply_bands(b, v[:, None])[:, 0],
        lambda s: eca_step(rule, s),
        state, T, burn, np.random.default_rng(seed))


def circulant_log_stretch(rule, v0, T, burn):
    """Exact ln(|J^T v0| / |J^burn v0|) / (T - burn) for a normal circulant J."""
    lam = eca_eigenvalues(rule, v0.size)
    coeff = np.fft.fft(v0)
    norm = lambda t: np.sqrt(np.sum(np.abs(coeff) ** 2 * np.abs(lam) ** (2 * t)) / v0.size)
    return (np.log(norm(T)) - np.log(norm(burn))) / (T - burn)


@pytest.mark.parametrize("rule", [150, 90, 60, 105])
def test_single_vector_matches_the_circulant_closed_form(rule):
    N, T, burn, seed = 101, 120, 40, 9
    state = ring(N, seed)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(N)
    mle, t_dead = eca_mle_1v(rule, state, T, burn, seed)
    assert t_dead is None
    assert mle == pytest.approx(circulant_log_stretch(rule, v0, T, burn), abs=1e-9)


@pytest.mark.parametrize("rule, tol", [(150, 0.02), (90, 0.02), (60, 0.02)])
def test_single_vector_is_close_to_the_exact_mle(rule, tol):
    # The k = 1 estimator converges to ln(sigma_max) like 1/T because the
    # vector is a mixture of Fourier modes whose stretch ratios approach 1
    # only as (2 pi / N)^2. At N = 101, T = 400, burn 200 that transient is
    # a few 1e-3; the tolerance is set once, with margin, and not loosened.
    mle, _ = eca_mle_1v(rule, ring(101, 10), 400, 200, 10)
    assert abs(mle - eca_mle(rule)) < tol


@pytest.mark.parametrize("rule", [0, 8, 32])
def test_annihilating_rules_give_minus_infinity(rule):
    mle, t_dead = eca_mle_1v(rule, ring(41, 11), 30, 5, 11)
    assert mle == -np.inf
    assert t_dead is not None and 1 <= t_dead <= 30


@pytest.mark.parametrize("rule", [30, 110, 54])
def test_single_vector_agrees_with_full_benettin_top_exponent(rule):
    # Same trajectory, same window; the two estimators differ by their
    # starting vector only, i.e. by a finite-time transient.
    N, T, burn = 101, 200, 100
    state = ring(N, 12)
    top = spectrum_along_trajectory(rule, state, T, burn)[0]
    mle, _ = eca_mle_1v(rule, state, T, burn, 12)
    assert abs(mle - top) < 0.03


def test_single_vector_rejects_bad_window():
    with pytest.raises(ValueError):
        eca_mle_1v(30, ring(41, 1), 10, 10, 1)


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------

def test_eca_wrapper_returns_flat_dict():
    out = eca_damage_and_mle(150, ring(43, 13), T=20, burn=5, window=5, rng=np.random.default_rng(13))
    assert set(out) >= {"v_front", "D_norm", "fill", "h_final", "r_final", "mle", "t_dead", "h", "r"}
    assert out["h"].shape == (21,) and out["v_front"] == pytest.approx(1.0)
    assert abs(out["mle"] - np.log(3)) < 0.1


def test_ot_wrapper_returns_flat_dict():
    state = np.random.default_rng(14).integers(0, 2, size=(13, 13))
    out = ot_damage_and_mle(PARITY_VN_INCLUSIVE, state, T=5, burn=1, window=2, rng=np.random.default_rng(14))
    assert out["v_front"] == pytest.approx(1.0)
    assert 0 < out["fill"] < 1 and out["h"].shape == (6,)
    assert np.isfinite(out["mle"])


def test_ot_wrapper_on_the_moore_neighbourhood():
    state = np.random.default_rng(15).integers(0, 2, size=(13, 13))
    out = ot_damage_and_mle(PARITY_MOORE_INCLUSIVE, state, T=5, burn=1, window=2,
                            rng=np.random.default_rng(15), neighbourhood=MOORE)
    assert out["v_front"] == pytest.approx(1.0)
    assert np.array_equal(out["h"][:4], [1, 9, 9, 25]) and out["D_norm"] < 1
    assert np.isfinite(out["mle"])
    # ln 9 is the exact exponent; at T = 5 on 13 x 13 only the order of magnitude holds.
    assert 1.5 < out["mle"] < np.log(9) + 0.1
    dead = ot_damage_and_mle(0, state, T=5, burn=1, window=2, rng=np.random.default_rng(15),
                             neighbourhood=MOORE)
    assert dead["mle"] == -np.inf and dead["t_dead"] == 1
