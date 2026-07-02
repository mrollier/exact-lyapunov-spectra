"""Claim C3: the maximal Lyapunov exponents take the stated exact values.

* ECAs: rules 150 and 105 attain ln 3 (the ECA maximum); rule 90 attains ln 2.
* 2-D parity: von Neumann ln 5, Moore ln 9, radius-2 von Neumann ln 13.
* The Moore value factorises as 2 ln 3 (the neighbourhood is a Cartesian product
  of two 1-D radius-one neighbourhoods, over which the MLE is additive).
"""
import numpy as np
import pytest

from lyapunov.spectra import (
    eca_mle,
    parity_2d_mle,
    VON_NEUMANN_2D,
    MOORE_2D,
    VON_NEUMANN_R2_2D,
)


def test_eca_mle_values():
    assert eca_mle(150) == pytest.approx(np.log(3))
    assert eca_mle(105) == pytest.approx(np.log(3))
    assert eca_mle(90) == pytest.approx(np.log(2))


def test_2d_parity_mle_values():
    assert parity_2d_mle(VON_NEUMANN_2D) == pytest.approx(np.log(5))
    assert parity_2d_mle(MOORE_2D) == pytest.approx(np.log(9))
    assert parity_2d_mle(VON_NEUMANN_R2_2D) == pytest.approx(np.log(13))


def test_moore_factorises_as_twice_ln3():
    assert parity_2d_mle(MOORE_2D) == pytest.approx(2 * np.log(3))
