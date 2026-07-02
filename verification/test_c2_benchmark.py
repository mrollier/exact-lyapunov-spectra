"""Claim C2: the closed-form affine spectrum matches a stable numerical routine
to floating-point precision, across a range of N and T (the benchmark).

Two independent numerical routines are used as the reference:
* direct multiplication in float64, which for a normal (circulant) Jacobian is
  exact at the top of the spectrum at any horizon;
* Benettin's QR algorithm, which converges to the whole spectrum as T grows and
  whose exponent sum equals ln|det J| at every T (see C6).
"""
import numpy as np
import pytest

from lyapunov.jacobian import eca_jacobian
from lyapunov.benettin import (
    closed_form_spectrum,
    direct_multiplication_spectrum,
    benettin_spectrum,
)


@pytest.mark.parametrize("rule", [15, 60, 90, 102, 150, 105])
@pytest.mark.parametrize("N", [31, 64, 101])
@pytest.mark.parametrize("T", [50, 200])
def test_direct_float64_matches_closed_form_at_the_top(rule, N, T):
    # The largest exponent (the MLE) is reproduced to machine precision.
    closed = closed_form_spectrum(rule, N)
    got = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float64)
    assert got[0] == pytest.approx(closed[0], abs=1e-10)


def test_benettin_converges_to_closed_form_across_rules():
    for rule, N in [(60, 31), (90, 41), (150, 25)]:
        closed = closed_form_spectrum(rule, N)
        # closed form may contain -inf (zero singular values); compare the finite part
        finite = np.isfinite(closed)
        ben = benettin_spectrum(eca_jacobian(rule, N), T=6000)
        assert np.max(np.abs(ben[finite] - closed[finite])) < 1e-2
