"""Claim C5 (Sec. 4.2): for the parity rule on a graph the Lyapunov spectrum is
``ln|lambda_k(A) + a_o|`` and the MLE is ``ln rho(A + a_o I)``, for both the
self-exclusive (``a_o = 0``) and the self-inclusive (``a_o = 1``) rule, and the
spectral radius of a connected graph is bounded below by its mean degree, so
the parity MLE is positive on every connected graph with mean degree above one.

Checked numerically on Watts-Strogatz and Barabasi-Albert networks (N = 200,
mean degree 6 / m = 3). The last test, on eigenvector centrality, is
supplementary material: it backs the network figure of the original submission,
which the resubmitted manuscript no longer contains.
"""
import numpy as np
import networkx as nx
import pytest

from lyapunov.parity import (
    parity_lyapunov_spectrum,
    parity_mle,
    eigenvector_centrality,
    long_time_amplitude_ratio,
)

WS = nx.to_numpy_array(nx.watts_strogatz_graph(200, 6, 0.2, seed=20240601), dtype=int)
BA = nx.to_numpy_array(nx.barabasi_albert_graph(200, 3, seed=20240601), dtype=int)


@pytest.mark.parametrize("A", [WS, BA])
@pytest.mark.parametrize("self_inclusive", [False, True])
def test_mle_equals_log_spectral_radius_of_shifted_adjacency(A, self_inclusive):
    # Eq. (15) and the line below it: MLE = ln rho(A + a_o I) = ln(lambda_N(A) + a_o),
    # the second equality because A is nonnegative (Perron-Frobenius).
    a_o = 1.0 if self_inclusive else 0.0
    eig = np.linalg.eigvalsh(A.astype(float))
    rho = np.max(np.abs(eig + a_o))
    assert parity_mle(A, self_inclusive) == pytest.approx(np.log(rho), abs=1e-9)
    assert rho == pytest.approx(eig.max() + a_o, abs=1e-9)
    spectrum = parity_lyapunov_spectrum(A, self_inclusive, drop_zeros=False)
    assert np.max(spectrum) == pytest.approx(parity_mle(A, self_inclusive), abs=1e-9)


@pytest.mark.parametrize("A", [WS, BA])
def test_spectral_radius_is_at_least_the_mean_degree(A):
    # Sec. 4.2, citing Hong (1993): rho(A) >= 2|E| / |V| for a connected graph,
    # the Rayleigh quotient of the all-ones vector; hence MLE >= ln(mean degree).
    G = nx.from_numpy_array(A)
    assert nx.is_connected(G)
    mean_degree = A.sum() / A.shape[0]
    rho = np.max(np.abs(np.linalg.eigvalsh(A.astype(float))))
    assert rho >= mean_degree - 1e-12
    assert parity_mle(A) >= np.log(mean_degree) - 1e-12 > 0


@pytest.mark.parametrize("A", [WS, BA])
def test_amplitude_proportional_to_eigenvector_centrality(A):
    centrality = eigenvector_centrality(A)
    ratio = long_time_amplitude_ratio(A, T=300)
    # long-time amplitude ratio converges to the (unit-norm) centrality
    assert np.allclose(ratio, centrality, atol=1e-6)
    # proportionality is exact: ratio_i / centrality_i is constant across nodes
    scale = ratio / centrality
    assert np.std(scale) < 1e-6
