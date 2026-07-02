"""Claim C5: for the parity rule on a graph the MLE equals ln(spectral radius),
and the per-node long-time perturbation amplitude is proportional to the node's
eigenvector centrality. Checked numerically on the manuscript's Watts-Strogatz
and Barabasi-Albert networks (N=200, mean degree 6 / m=3).
"""
import numpy as np
import networkx as nx
import pytest

from lyapunov.parity import (
    parity_mle,
    eigenvector_centrality,
    long_time_amplitude_ratio,
)

WS = nx.to_numpy_array(nx.watts_strogatz_graph(200, 6, 0.2, seed=20240601), dtype=int)
BA = nx.to_numpy_array(nx.barabasi_albert_graph(200, 3, seed=20240601), dtype=int)


@pytest.mark.parametrize("A", [WS, BA])
def test_mle_equals_log_spectral_radius(A):
    rho = np.max(np.abs(np.linalg.eigvalsh(A.astype(float))))
    assert parity_mle(A) == pytest.approx(np.log(rho), abs=1e-9)


@pytest.mark.parametrize("A", [WS, BA])
def test_amplitude_proportional_to_eigenvector_centrality(A):
    centrality = eigenvector_centrality(A)
    ratio = long_time_amplitude_ratio(A, T=300)
    # long-time amplitude ratio converges to the (unit-norm) centrality
    assert np.allclose(ratio, centrality, atol=1e-6)
    # proportionality is exact: ratio_i / centrality_i is constant across nodes
    scale = ratio / centrality
    assert np.std(scale) < 1e-6
