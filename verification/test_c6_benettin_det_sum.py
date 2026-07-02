"""Claim C6 (Benettin sanity check): the sum of all Lyapunov exponents equals
ln|det J| at every horizon T.

This holds exactly for Benettin's algorithm because the product of the diagonal
QR stretching factors at each step is |det J|, so the accumulated sum telescopes
to T ln|det J| and the time average is ln|det J| for any T.
"""
import numpy as np
import networkx as nx
import pytest

from lyapunov.jacobian import eca_jacobian
from lyapunov.parity import parity_jacobian
from lyapunov.benettin import benettin_spectrum


@pytest.mark.parametrize("rule, N", [(150, 25), (90, 41), (60, 31), (105, 20)])
@pytest.mark.parametrize("T", [1, 3, 17, 100])
def test_exponent_sum_equals_log_det_eca(rule, N, T):
    J = eca_jacobian(rule, N)
    det = np.linalg.det(J.astype(float))
    if abs(det) < 1e-9:
        pytest.skip("singular Jacobian: ln|det J| = -inf")
    spec = benettin_spectrum(J, T)
    assert np.sum(spec) == pytest.approx(np.log(abs(det)), abs=1e-8)


@pytest.mark.parametrize("T", [1, 10, 50])
def test_exponent_sum_equals_log_det_parity_graph(T):
    A = nx.to_numpy_array(nx.watts_strogatz_graph(40, 6, 0.2, seed=11), dtype=int)
    J = parity_jacobian(A, self_inclusive=True)  # +I keeps det away from zero
    det = np.linalg.det(J.astype(float))
    if abs(det) < 1e-9:
        pytest.skip("singular Jacobian")
    spec = benettin_spectrum(J, T)
    assert np.sum(spec) == pytest.approx(np.log(abs(det)), abs=1e-8)
