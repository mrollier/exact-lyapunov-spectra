"""Claim C7: reproduce the numerical artefacts of naive computations and show the
correct result beside them. The framing is factual: the naive routines are not
wrong in principle, they fail in specific, reproducible regimes.

Three regimes are demonstrated, with an honest account of each:

1. Integer *magnitude* overflow. ``numpy.linalg.matrix_power`` on an int64
   adjacency matrix overflows for moderate exponents, so the walk counts
   ``A^t`` come out with the wrong magnitude (indeed negative). The object-int
   power is exact.

2. Parity via the floating-point route. Computing ``A^t`` in float64 and then
   reducing modulo two gives the *wrong* defect pattern, because float64 loses
   integer precision above 2**53 and the low bit is destroyed. The GF(2) power
   is correct.

   (Note: the *pure int64* route ``matrix_power(int64) % 2`` happens to preserve
   parity, because two's-complement wraparound is modulo 2**64 and 2 divides
   2**64. The danger is therefore the floating-point path, which is how matrix
   powers are most often formed. The GF(2) routine is safe either way and never
   overflows, so it is what the defect-pattern code uses.)

3. Floating-point precision in the spectrum. Direct multiplication in float16
   loses the slow-growing directions of the Lyapunov spectrum, while the exact
   closed form retains them.
"""
import numpy as np
import networkx as nx

from lyapunov.gf2 import int_matrix_power, gf2_matrix_power
from lyapunov.jacobian import eca_jacobian
from lyapunov.benettin import closed_form_spectrum, direct_multiplication_spectrum


def test_int64_matrix_power_overflows_walk_counts():
    # Complete graph on 12 nodes: A^t entries ~ (n-1)^t exceed int64 for t=25.
    A = nx.to_numpy_array(nx.complete_graph(12), dtype=int)
    t = 25
    naive = np.linalg.matrix_power(A.astype(np.int64), t)
    exact = int_matrix_power(A, t)
    # exact count is astronomically large; naive int64 has wrapped to negative
    assert int(exact[0, 0]) > np.iinfo(np.int64).max
    assert naive[0, 0] < 0                       # overflow artefact
    assert int(naive[0, 0]) != int(exact[0, 0])  # wrong magnitude


def test_float_route_corrupts_parity_but_gf2_is_correct():
    A = nx.to_numpy_array(nx.complete_graph(12), dtype=int)
    t = 25
    exact_parity = (int_matrix_power(A, t) % 2).astype(np.int64)

    # Naive float64 power then mod 2: precision loss corrupts the parity.
    float_parity = (np.linalg.matrix_power(A.astype(np.float64), t) % 2).astype(np.int64)
    assert not np.array_equal(float_parity, exact_parity)   # artefact present

    # GF(2) power is exact.
    assert np.array_equal(gf2_matrix_power(A, t), exact_parity)


def test_float16_direct_multiplication_loses_slow_directions():
    rule, N, T = 150, 101, 200
    closed = closed_form_spectrum(rule, N)
    f16 = direct_multiplication_spectrum(eca_jacobian(rule, N), T, np.float16)
    assert closed.min() < -3.0                   # exact spectrum reaches deep
    assert f16.min() > closed.min() + 1.0        # float16 plateaus (directions lost)
    assert abs(f16.max() - closed.max()) < 1e-3  # but the MLE is still captured
