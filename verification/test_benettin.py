"""Unit tests for the reference numerical Lyapunov routines (benettin.py).

These encode the paper's benchmark claims:
* Benettin's QR algorithm converges to the exact closed-form spectrum.
* The sum of the Benettin exponents equals ln|det J| at *every* horizon (C6).
* Direct multiplication in float64 is exact for a normal Jacobian (its strength
  at the top of the spectrum), while in float16 it loses the slow-growing
  directions to floating-point error -- the artefact the paper demonstrates.
"""
import numpy as np
import pytest

from lyapunov.jacobian import eca_jacobian
from lyapunov.spectra import eca_singular_values
from lyapunov.benettin import (
    benettin_spectrum,
    direct_multiplication_spectrum,
    closed_form_spectrum,
)


def _closed(rule, N):
    return np.sort(np.log(eca_singular_values(rule, N)))[::-1]


def test_benettin_sum_equals_log_det_at_every_T():
    # C6: sum of exponents == ln|det J| exactly, independent of T.
    J = eca_jacobian(150, 25)  # N=25 not divisible by 3 => det != 0
    logdet = np.log(np.abs(np.linalg.det(J.astype(float))))
    for T in [1, 5, 20, 100]:
        spec = benettin_spectrum(J, T)
        assert np.sum(spec) == pytest.approx(logdet, abs=1e-8)


def test_benettin_converges_to_closed_form():
    rule, N = 150, 25
    closed = _closed(rule, N)
    J = eca_jacobian(rule, N)
    err_small_T = np.max(np.abs(benettin_spectrum(J, 50) - closed))
    err_large_T = np.max(np.abs(benettin_spectrum(J, 4000) - closed))
    assert err_large_T < err_small_T          # it is converging
    assert err_large_T < 5e-3                  # and close at the horizon


def test_direct_multiplication_exact_at_top_all_precisions():
    # C2 / benchmark, top of the spectrum: the largest exponent (the MLE) is
    # recovered by direct multiplication at *every* precision, more exactly as the
    # precision increases ("direct multiplication is exact even at 16-bit").
    rule, N, T = 150, 101, 200
    closed = _closed(rule, N)
    f16 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float16)
    f32 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float32)
    f64 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float64)
    assert f64[0] == pytest.approx(closed[0], abs=1e-12)  # machine precision
    assert f32[0] == pytest.approx(closed[0], abs=1e-7)
    assert f16[0] == pytest.approx(closed[0], abs=1e-3)


def test_direct_multiplication_accurate_range_deepens_with_precision():
    # Bottom of the spectrum: direct multiplication plateaus, and the plateau
    # sits lower (more accurate) as precision increases -- but none of the direct
    # methods reaches the true minimum at T=200 (the slow directions are lost).
    rule, N, T = 150, 101, 200
    closed = _closed(rule, N)
    f16 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float16)
    f32 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float32)
    f64 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float64)
    assert f64.min() < f32.min() < f16.min()      # deeper accurate range
    assert f64.min() > closed.min() + 0.5          # yet still far from the truth


def test_benettin_recovers_bottom_where_direct_fails():
    # The roles reverse across the spectrum at horizon T=200: Benettin recovers
    # the smallest exponent while direct multiplication (even float64) loses it;
    # at the top Benettin has not yet converged while direct multiplication is exact.
    rule, N, T = 150, 101, 200
    closed = _closed(rule, N)
    ben = benettin_spectrum(eca_jacobian(rule, N), T)
    f64 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float64)
    assert ben.min() == pytest.approx(closed.min(), abs=3e-2)  # Benettin recovers bottom
    assert abs(f64.min() - closed.min()) > 1.0                 # direct loses bottom
    assert f64.max() == pytest.approx(closed.max(), abs=1e-12)  # direct exact at top
    assert abs(ben.max() - closed.max()) > 1e-3                # Benettin slow at top


def test_direct_multiplication_does_not_overflow_or_nan():
    # Scalar rescaling must keep the computation finite even in float16 at T=200,
    # where an unscaled 3**200 would overflow immediately.
    J = eca_jacobian(150, 101)
    for dtype in (np.float16, np.float32, np.float64):
        spec = direct_multiplication_spectrum(J, T=200, dtype=dtype)
        assert np.all(np.isfinite(spec))


def test_float16_loses_the_smallest_exponents():
    # C7 in miniature: the float16 direct-multiplication spectrum plateaus at the
    # bottom -- its smallest exponents sit well above the exact ones, and above
    # even the float64 plateau.
    rule, N, T = 150, 101, 200
    closed = _closed(rule, N)
    f16 = direct_multiplication_spectrum(eca_jacobian(rule, N), T=T, dtype=np.float16)
    f64 = direct_multiplication_spectrum(eca_jacobian(rule, N), T=T, dtype=np.float64)
    assert f16.min() > f64.min() + 0.05        # float16 plateau is shallower
    assert f16.min() > closed.min() + 0.05     # and both miss the true minimum


def test_closed_form_spectrum_helper_sorted_descending():
    spec = closed_form_spectrum(150, 51)
    assert np.all(np.diff(spec) <= 1e-12)  # non-increasing
    assert spec[0] == pytest.approx(np.log(3), abs=1e-9)
