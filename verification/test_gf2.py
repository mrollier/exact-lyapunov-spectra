"""Unit tests for the int64-safe matrix-power core (src/lyapunov/gf2.py).

These guard the central numerical hazard of the paper: ``numpy.linalg.matrix_power``
overflows silently on integer arrays for moderate exponents. The functions here
must give the *exact* answer instead.
"""
import numpy as np
import pytest

from lyapunov.gf2 import int_matrix_power, gf2_matrix_power, gf2_matmul


def test_int_power_matches_numpy_when_no_overflow():
    # Small exponent, values stay well within int64 -> must agree with numpy.
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=object)
    for t in range(0, 6):
        expected = np.linalg.matrix_power(A.astype(np.int64), t)
        got = int_matrix_power(A, t)
        assert np.array_equal(got.astype(np.int64), expected)


def test_int_power_is_exact_where_int64_overflows():
    # J = all-ones 2x2 has the closed form J^t = 2^(t-1) * J.
    # At t = 70, 2^69 far exceeds int64_max (2^63 - 1), so numpy overflows
    # while our object-dtype power stays exact.
    J = np.ones((2, 2), dtype=object)
    t = 70
    exact = 2 ** (t - 1)
    got = int_matrix_power(J, t)
    assert got[0, 0] == exact  # exact big Python int, no wraparound

    # Demonstrate the hazard: the naive int64 route disagrees (it overflows).
    naive = np.linalg.matrix_power(np.ones((2, 2), dtype=np.int64), t)
    assert int(naive[0, 0]) != exact


def test_gf2_power_equals_int_power_mod_2():
    # Reducing the exact integer power mod 2 must equal the GF(2) power,
    # for a range of exponents including large ones.
    A = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]], dtype=object)
    for t in [0, 1, 2, 3, 7, 20, 65, 128]:
        exact_mod2 = (int_matrix_power(A, t) % 2).astype(np.int64)
        gf2 = gf2_matrix_power(A.astype(np.int64), t)
        assert np.array_equal(gf2, exact_mod2)


def test_power_zero_is_identity():
    A = np.array([[1, 1], [0, 1]], dtype=object)
    assert np.array_equal(int_matrix_power(A, 0).astype(np.int64), np.eye(2, dtype=np.int64))
    assert np.array_equal(gf2_matrix_power(A.astype(np.int64), 0), np.eye(2, dtype=np.int64))


def test_gf2_matmul_reduces_mod_2():
    A = np.array([[1, 1], [1, 1]], dtype=np.int64)
    # (A @ A) over the integers is [[2,2],[2,2]] -> mod 2 -> zeros.
    assert np.array_equal(gf2_matmul(A, A), np.zeros((2, 2), dtype=np.int64))


def test_gf2_output_is_binary():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=(6, 6))
    P = gf2_matrix_power(A, 13)
    assert set(np.unique(P)).issubset({0, 1})


def test_negative_exponent_rejected():
    A = np.eye(3, dtype=object)
    with pytest.raises(ValueError):
        int_matrix_power(A, -1)
    with pytest.raises(ValueError):
        gf2_matrix_power(np.eye(3, dtype=np.int64), -1)
