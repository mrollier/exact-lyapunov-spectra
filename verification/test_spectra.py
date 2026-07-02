"""Unit tests for the closed-form spectra (spectra.py) and Jacobians (jacobian.py).

The strongest tests here cross-validate two independent computations: the
closed-form singular values (a discrete Fourier transform of the gradient
stencil) against a direct SVD of the explicitly built constant Jacobian. If they
agree, both the Jacobian construction and the closed form are correct.
"""
import numpy as np
import pytest

from lyapunov.jacobian import eca_jacobian, build_torus_parity_jacobian
from lyapunov.spectra import (
    eca_singular_values,
    eca_lyapunov_spectrum,
    eca_mle,
    parity_2d_singular_values,
    parity_2d_mle,
    structure_factor,
    VON_NEUMANN_2D,
    MOORE_2D,
    VON_NEUMANN_R2_2D,
)


@pytest.mark.parametrize("rule", [0, 15, 51, 60, 90, 102, 150, 105, 165])
@pytest.mark.parametrize("N", [7, 12, 25])
def test_eca_closed_form_matches_direct_svd(rule, N):
    sv_closed = np.sort(eca_singular_values(rule, N))
    J = eca_jacobian(rule, N)
    sv_direct = np.sort(np.linalg.svd(J.astype(float), compute_uv=False))
    assert np.allclose(sv_closed, sv_direct, atol=1e-10)


def test_eca_mle_values_C3():
    # Rules 150 and 105 attain ln 3 (the ECA maximum); rule 90 attains ln 2.
    assert eca_mle(150) == pytest.approx(np.log(3))
    assert eca_mle(105) == pytest.approx(np.log(3))
    assert eca_mle(90) == pytest.approx(np.log(2))
    assert eca_mle(60) == pytest.approx(np.log(2))
    assert eca_mle(15) == pytest.approx(np.log(1))  # == 0
    assert eca_mle(0) == -np.inf  # zero gradient: no growth direction


def test_eca_mle_equals_max_of_spectrum():
    for rule in [15, 60, 90, 150]:
        spec = eca_lyapunov_spectrum(rule, 301)
        assert np.max(spec) == pytest.approx(eca_mle(rule), abs=1e-9)


def test_rule150_closed_form_matches_analytic_expression():
    # Lambda_k = ln|1 + 2 cos(2 pi k / N)| for rule 150.
    N = 101
    k = np.arange(N)
    expected = np.log(np.abs(1 + 2 * np.cos(2 * np.pi * k / N)))
    got = np.log(eca_singular_values(150, N))
    assert np.allclose(np.sort(got), np.sort(expected), atol=1e-10)


def test_structure_factor_matches_table2_formulas():
    N = 37
    rng = np.random.default_rng(1)
    for _ in range(20):
        k, l = rng.integers(0, N, size=2)
        a, b = 2 * np.pi * k / N, 2 * np.pi * l / N
        vn = 2 * np.cos(a) + 2 * np.cos(b)
        moore = vn + 4 * np.cos(a) * np.cos(b)
        r2 = moore + 2 * np.cos(2 * a) + 2 * np.cos(2 * b)
        assert structure_factor(VON_NEUMANN_2D, k, l, N) == pytest.approx(vn)
        assert structure_factor(MOORE_2D, k, l, N) == pytest.approx(moore)
        assert structure_factor(VON_NEUMANN_R2_2D, k, l, N) == pytest.approx(r2)


def test_parity_2d_mle_values_C3():
    # Self-inclusive parity MLEs: ln 5, ln 9, ln 13; Moore factorises as 2 ln 3.
    assert parity_2d_mle(VON_NEUMANN_2D) == pytest.approx(np.log(5))
    assert parity_2d_mle(MOORE_2D) == pytest.approx(np.log(9))
    assert parity_2d_mle(VON_NEUMANN_R2_2D) == pytest.approx(np.log(13))
    assert parity_2d_mle(MOORE_2D) == pytest.approx(2 * np.log(3))


@pytest.mark.parametrize("offsets", [VON_NEUMANN_2D, MOORE_2D, VON_NEUMANN_R2_2D])
def test_parity_2d_closed_form_matches_direct_svd(offsets):
    # Cross-check the 2-D DFT closed form against the SVD of the block-circulant
    # Jacobian on a small torus.
    L = 6
    sigma_grid = parity_2d_singular_values(offsets, L)
    sv_closed = np.sort(sigma_grid.ravel())
    J = build_torus_parity_jacobian(offsets, L)
    sv_direct = np.sort(np.linalg.svd(J.astype(float), compute_uv=False))
    assert np.allclose(sv_closed, sv_direct, atol=1e-9)


def test_eca_jacobian_is_circulant_with_gradient():
    # Rule 90 gradient (1,0,1): each row carries a_- (left), a_o (centre), a_+.
    J = eca_jacobian(90, 5)
    assert J[0, 0] == 0  # a_o
    assert J[0, 1] == 1  # a_+
    assert J[0, 4] == 1  # a_- wraps around
    assert J.sum() == 5 * 2  # two ones per row


def test_eca_jacobian_rejects_nonaffine():
    with pytest.raises(ValueError):
        eca_jacobian(30, 10)
