"""Unit tests for the convergence-study routines added to benettin.py.

These encode the calibration facts demonstrated in
``notebooks/03_benettin_convergence.ipynb`` (the replacement for Fig. 3):

* ``benettin_running_average`` agrees with ``benettin_spectrum`` at its final
  horizon, keeps the exponent sum equal to ln|det J| at every checkpoint (C6),
  is exact from the first step when started in the eigenbasis of a normal J,
  and converges faster at the ends of the spectrum after a burn-in.
* ``direct_multiplication_unscaled`` is the method of Vispoel et al. (2024)
  as stated: exact at the top, resolving fewer exponents as T grows, and with
  no result at all once sigma_max^(2T) exceeds the largest float64.
* ``precision_floor`` is the closed-form floor ln(sigma_max) + ln(eps)/(2T).

BLAS threading changes the rounding pathway of QR and hence the last digits of
the slowest-converging exponents; every tolerance below is chosen to be
insensitive to that.
"""
import numpy as np
import pytest

from lyapunov.jacobian import eca_jacobian
from lyapunov.benettin import (
    benettin_spectrum,
    benettin_running_average,
    closed_form_spectrum,
    direct_multiplication_unscaled,
    precision_floor,
)

RULE, N = 150, 101


@pytest.fixture(scope="module")
def J():
    return eca_jacobian(RULE, N)


@pytest.fixture(scope="module")
def exact():
    return closed_form_spectrum(RULE, N)


# --------------------------------------------------------------- Benettin
def test_running_average_matches_benettin_spectrum_at_final_horizon(J):
    T = 60
    single = benettin_spectrum(J, T)
    running = benettin_running_average(J, T)  # default checkpoints = {T}
    assert set(running) == {T}
    assert np.allclose(np.sort(running[T])[::-1], single, atol=1e-12)


def test_running_average_sum_is_log_det_at_every_checkpoint():
    J = eca_jacobian(RULE, 25)  # N = 25 not divisible by 3 => det != 0
    logdet = np.log(abs(np.linalg.det(J.astype(float))))
    out = benettin_running_average(J, 100, burn=7, checkpoints=[1, 5, 20, 100])
    for T, spec in out.items():
        assert np.sum(spec) == pytest.approx(logdet, abs=1e-8), T


def test_running_average_rejects_bad_arguments(J):
    with pytest.raises(ValueError):
        benettin_running_average(J, 0)
    with pytest.raises(ValueError):
        benettin_running_average(J, 10, burn=-1)
    with pytest.raises(ValueError):
        benettin_running_average(J, 10, checkpoints=[11])
    with pytest.raises(ValueError):
        benettin_running_average(J, 10, Q0=np.eye(3))


def test_eigenbasis_start_is_exact_from_the_first_step(J, exact):
    # J is symmetric (rule 150 has gradient (1, 1, 1)), hence normal: from its
    # eigenbasis R is diagonal at every step and the average is exact at T = 1.
    w, X = np.linalg.eigh(J.astype(float))
    order = np.argsort(-np.abs(w))
    out = benettin_running_average(J, 5, checkpoints=[1, 5], Q0=X[:, order])
    assert np.max(np.abs(out[1] - exact)) < 1e-12
    assert np.max(np.abs(out[5] - exact)) < 1e-12


def test_identity_start_error_decays_like_one_over_T(J, exact):
    # Frame-alignment transient in the running average: a log-log slope close
    # to -1 for the top and bottom exponents. The bottom halves exactly with
    # each doubling of T from the start; the top, whose gap ratio 0.9987 is the
    # slowest to resolve, approaches the halving from below (1.8 -> 2.0).
    Ts = [200, 400, 800, 1600, 3200]
    out = benettin_running_average(J, Ts[-1], checkpoints=Ts)
    for k, slope_tol in ((0, 0.15), (N - 1, 0.01)):
        err = np.array([abs(out[T][k] - exact[k]) for T in Ts])
        slope = np.polyfit(np.log(Ts), np.log(err), 1)[0]
        assert abs(slope + 1.0) < slope_tol, (k, slope)
        ratios = err[:-1] / err[1:]
        assert np.all(ratios > 1.7) and abs(ratios[-1] - 2.0) < 0.05, (k, ratios)


def test_burn_in_makes_the_ends_exact_but_not_the_middle(J, exact):
    T = 200
    out = benettin_running_average(J, T, burn=200)[T]
    err = np.abs(out - exact)
    assert err[0] < 1e-3            # top: aligned after the burn-in
    assert err[-1] < 1e-10          # bottom: aligned to machine precision
    assert 2e-2 < err[N // 2] < 5e-2  # median: alignment rate ~0.9994/step


def test_burn_in_removes_the_transient_at_the_ends(J, exact):
    T = 200
    plain = benettin_running_average(J, T)[T]
    burnt = benettin_running_average(J, T, burn=200)[T]
    for k in (0, N - 1):
        assert abs(burnt[k] - exact[k]) < 0.2 * abs(plain[k] - exact[k])


# --------------------------------------------------- direct multiplication
def test_unscaled_direct_method_is_exact_at_the_top(J, exact):
    for T in (50, 200, 323):
        got = direct_multiplication_unscaled(J, T)
        assert got[0] == pytest.approx(exact[0], abs=1e-12)
        assert np.all(got[np.isfinite(got)] <= exact[0] + 1e-9)


def test_unscaled_direct_method_resolves_fewer_exponents_as_T_grows(J):
    counts = []
    for T in (50, 100, 200, 300):
        spec = direct_multiplication_unscaled(J, T)
        counts.append(int((spec > precision_floor(3.0, T) + 0.02).sum()))
    assert counts == [31, 21, 15, 11]


def test_unscaled_direct_method_overflows_beyond_T_323(J):
    # sigma_max = 3: 3^(2T) exceeds the largest float64 for T >= 324.
    T_max = int(np.log(np.finfo(np.float64).max) / (2 * np.log(3.0)))
    assert T_max == 323
    spec = direct_multiplication_unscaled(J, T_max)          # nan for negative eigenvalues, never inf
    assert np.isfinite(spec[0]) and not np.any(np.isinf(spec))
    with pytest.raises(OverflowError):
        direct_multiplication_unscaled(J, T_max + 1)


def test_unscaled_direct_method_rejects_T_below_one(J):
    with pytest.raises(ValueError):
        direct_multiplication_unscaled(J, 0)


# ------------------------------------------------------------- the floor
def test_precision_floor_formula_and_monotonicity():
    eps = np.finfo(np.float64).eps
    assert precision_floor(3.0, 100) == pytest.approx(np.log(3) + np.log(eps) / 200)
    floors = [precision_floor(3.0, T) for T in (50, 100, 200, 300, 500)]
    assert np.all(np.diff(floors) > 0)          # the floor RISES with T
    assert floors[-1] < np.log(3)               # but stays below the MLE
    assert precision_floor(3.0, 500) == pytest.approx(1.0626, abs=1e-3)
    assert precision_floor(2.0, 500) == pytest.approx(0.6571, abs=1e-3)
    with pytest.raises(ValueError):
        precision_floor(3.0, 0)


def test_reported_exponents_plateau_at_the_floor_or_are_nan(J, exact):
    # Below the floor the eigenvalues of Y Y^T are rounding noise on both
    # sides of zero. Every exponent whose exact value lies more than 0.1 below
    # the floor is therefore either reported at the floor (positive noise,
    # within 0.06) or not reported at all (negative noise, nan). None is
    # reported correctly, and both outcomes are common.
    for T in (50, 100, 200, 300):
        spec = direct_multiplication_unscaled(J, T)
        floor = precision_floor(3.0, T)
        below = exact < floor - 0.1
        assert below.sum() > 50
        reported = spec[below]
        is_nan = np.isnan(reported)
        assert np.all(np.abs(reported[~is_nan] - floor) < 0.06)
        assert (~is_nan).sum() > 20 and is_nan.sum() > 20
        assert np.all(np.abs(reported[~is_nan] - exact[below][~is_nan]) > 0.05)
        assert not np.any(np.isnan(spec[~below]))         # resolved exponents are always numbers
