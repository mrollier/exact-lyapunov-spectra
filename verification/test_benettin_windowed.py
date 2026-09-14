"""Unit tests for the windowed Benettin estimator (the revised Fig. 3).

One QR run of 4000 steps on rule 150, N = 101, from Q0 = I stores the per-step
log stretches; every estimator is a slice. The tests encode:

* consistency: the cumulative slice equals ``benettin_spectrum``; row sums are
  ln|det J| at every step (C6); a window with no burn-in is the running average.
* the reference table of ``figures/make_convergence_figure.py`` (W = 1000):
  fractions within 1e-3 and 1e-2 to 0.02, maximum errors to a factor of 2.
* the two slow bands: at B = 300, W = 1000 every exponent outside k = 1..15 and
  k = 50..66 is within 1e-6 (not machine precision: errors between 1e-12 and
  1e-6 persist up to k = 31 and at scattered k up to 87).
* the maximum error falls by orders of magnitude with B, but not strictly
  monotonically: from B = 100 to 300 it rises by 0.5 % (1.772e-2 to 1.782e-2),
  and per k the error rises between consecutive grid values 37 times out of
  505, all in the interior band where nearly degenerate pairs swap.
* the figure's two same-budget estimates: burn-in 100 then a 100-step window
  beats no burn-in with a 200-step average on every summary number.
"""
import numpy as np
import pytest

from lyapunov.jacobian import eca_jacobian
from lyapunov.benettin import (
    benettin_spectrum,
    benettin_log_stretch,
    closed_form_spectrum,
    cumulative_spectrum,
    windowed_spectrum,
)

RULE, N, TT, W = 150, 101, 4000, 1000

REFERENCE = {   # estimator: (fraction within 1e-3, fraction within 1e-2, max error)
    "running_4000": (0.09, 1.00, 6.9e-3),
    0: (0.02, 0.45, 2.8e-2),
    100: (0.79, 0.97, 1.8e-2),
    300: (0.89, 0.98, 1.8e-2),
    1000: (1.00, 1.00, 6.5e-5),
    3000: (1.00, 1.00, 4.0e-7),
}


@pytest.fixture(scope="module")
def J():
    return eca_jacobian(RULE, N)


@pytest.fixture(scope="module")
def exact():
    return closed_form_spectrum(RULE, N)


@pytest.fixture(scope="module")
def L(J):
    return benettin_log_stretch(J, TT)


def _summary(est, exact):
    e = np.abs(est - exact)
    return float(np.mean(e < 1e-3)), float(np.mean(e < 1e-2)), float(e.max())


def test_shape_and_row_sums_are_log_det(J, L):
    assert L.shape == (TT, N)
    logdet = np.linalg.slogdet(J.astype(float))[1]
    assert np.allclose(L.sum(axis=1), logdet, atol=1e-10)


def test_cumulative_slice_equals_benettin_spectrum(J, L):
    for T in (1, 50, 200):
        assert np.allclose(cumulative_spectrum(L, T), benettin_spectrum(J, T), atol=1e-12)
    assert np.array_equal(windowed_spectrum(L, 0, 200), cumulative_spectrum(L, 200))


def test_starting_frame_and_bad_arguments(J, L):
    w, X = np.linalg.eigh(J.astype(float))
    order = np.argsort(-np.abs(w))
    L_eig = benettin_log_stretch(J, 3, Q0=X[:, order])
    assert np.allclose(np.sort(L_eig[0])[::-1], np.log(np.abs(w[order])), atol=1e-12)
    with pytest.raises(ValueError):
        benettin_log_stretch(J, 0)
    with pytest.raises(ValueError):
        benettin_log_stretch(J, 2, Q0=np.eye(3))
    with pytest.raises(ValueError):
        windowed_spectrum(L, -1, 10)
    with pytest.raises(ValueError):
        windowed_spectrum(L, 0, 0)
    with pytest.raises(ValueError):
        windowed_spectrum(L, TT - 5, 10)


def test_reference_table(L, exact):
    got = {"running_4000": _summary(cumulative_spectrum(L, TT), exact)}
    for B in (0, 100, 300, 1000, 3000):
        got[B] = _summary(windowed_spectrum(L, B, W), exact)
    for key, (f3, f2, emax) in REFERENCE.items():
        g3, g2, gmax = got[key]
        assert abs(g3 - f3) <= 0.02, (key, g3)
        assert abs(g2 - f2) <= 0.02, (key, g2)
        assert emax / 2 <= gmax <= emax * 2, (key, gmax)


def test_two_slow_bands(L, exact):
    e = np.abs(windowed_spectrum(L, 300, W) - exact)
    k = np.arange(1, N + 1)
    slow = k[e > 1e-6]
    assert np.all((slow <= 15) | ((slow >= 50) & (slow <= 66)))
    assert np.any(slow <= 15) and np.any(slow >= 50)
    assert np.all(e[(k > 15) & (k < 50)] < 1e-6)
    assert np.all(e[k > 66] < 1e-6)
    assert not np.all(e[(k > 15) & (k < 50)] < 1e-12)      # not yet machine precision there


def test_windowed_error_falls_with_burn_in(L, exact):
    grid = (0, 30, 100, 300, 1000, 3000)
    worst = np.array([np.abs(windowed_spectrum(L, B, W) - exact).max() for B in grid])
    assert np.all(worst[1:] <= worst[:-1] * 1.01)        # B = 100 -> 300 rises by 0.5 %
    assert worst[3] / worst[4] > 100 and worst[4] / worst[5] > 100
    assert np.all(np.abs(windowed_spectrum(L, 1000, W) - exact) < 1e-3)


def test_same_budget_burn_in_beats_running_average(L, exact):
    # The figure's bottom panel: 200 steps either way.
    f3_b, f2_b, emax_b = _summary(windowed_spectrum(L, 100, 100), exact)
    f3_r, f2_r, emax_r = _summary(cumulative_spectrum(L, 200), exact)
    assert f3_b > f3_r and f2_b > f2_r and emax_b < emax_r
    assert f2_b > 0.8 and f2_r < 0.1          # measured: 0.85 and 0.04
