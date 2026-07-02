"""Unit tests for the parity rule on graphs (parity.py) -- claim C5 and links.

The parity rule's Jacobian is the adjacency matrix (plus the identity when
self-inclusive), so:
* its Lyapunov spectrum is the logarithm of the absolute adjacency spectrum;
* its MLE is the logarithm of the spectral radius;
* the long-time amplitude of a single-site perturbation is proportional to the
  seeded node's eigenvector centrality;
* the true defect pattern is A^t e_j (mod 2) -- walk-counting modulo two, which
  on a ring reproduces the Sierpinski triangle.
"""
import numpy as np
import networkx as nx
import pytest

from lyapunov.parity import (
    parity_jacobian,
    parity_lyapunov_spectrum,
    parity_mle,
    eigenvector_centrality,
    long_time_amplitude_ratio,
    defect_pattern,
)
from lyapunov.gf2 import gf2_matrix_power
from lyapunov.spectra import eca_singular_values


def _adj(G):
    return nx.to_numpy_array(G, dtype=int)


def test_spectrum_is_log_absolute_adjacency_spectrum():
    G = nx.watts_strogatz_graph(60, 6, 0.2, seed=7)
    A = _adj(G)
    expected = np.sort(np.log(np.abs(np.linalg.eigvalsh(A.astype(float)))[np.abs(np.linalg.eigvalsh(A.astype(float))) > 0]))[::-1]
    got = np.sort(parity_lyapunov_spectrum(A))[::-1]
    assert np.allclose(got, expected, atol=1e-9)


def test_mle_is_log_spectral_radius():
    for seed in (1, 2):
        G = nx.barabasi_albert_graph(100, 3, seed=seed)
        A = _adj(G)
        rho = np.max(np.abs(np.linalg.eigvalsh(A.astype(float))))
        assert parity_mle(A) == pytest.approx(np.log(rho), abs=1e-9)


def test_ring_parity_matches_eca_rule90_and_rule150():
    # A degree-2 ring is exactly ECA rule 90 (self-exclusive) / 150 (self-inclusive).
    N = 31
    A = _adj(nx.cycle_graph(N))
    exc = np.sort(parity_lyapunov_spectrum(A, self_inclusive=False))[::-1]
    inc = np.sort(parity_lyapunov_spectrum(A, self_inclusive=True))[::-1]
    r90 = np.sort(np.log(eca_singular_values(90, N)))[::-1]
    r150 = np.sort(np.log(eca_singular_values(150, N)))[::-1]
    assert np.allclose(exc, r90, atol=1e-9)
    assert np.allclose(inc, r150, atol=1e-9)


@pytest.mark.parametrize(
    "G", [
        nx.watts_strogatz_graph(200, 6, 0.2, seed=42),
        nx.barabasi_albert_graph(200, 3, seed=42),
    ],
)
def test_amplitude_proportional_to_eigenvector_centrality_C5(G):
    A = _adj(G)
    centrality = eigenvector_centrality(A)
    ratio = long_time_amplitude_ratio(A, T=200)
    # Both are unit-norm; at long times the ratio converges to the centrality.
    assert np.allclose(ratio, centrality, atol=1e-6)
    # And they agree with networkx's own eigenvector centrality (up to scale).
    nx_cent = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    nx_cent = nx_cent / np.linalg.norm(nx_cent)
    assert np.corrcoef(centrality, nx_cent)[0, 1] > 0.9999


def test_growth_rate_converges_to_mle():
    # Independent check: iterate the perturbation in tangent space with periodic
    # renormalisation (a power iteration) and accumulate the log-growth. The
    # time-averaged growth rate converges to the MLE, read purely from dynamics.
    G = nx.barabasi_albert_graph(200, 3, seed=3)
    A = _adj(G)
    J = parity_jacobian(A).astype(float)
    v = np.zeros(A.shape[0])
    v[0] = 1.0
    # Burn in so v aligns with the principal eigenvector, then measure the rate.
    for _ in range(300):
        v = J @ v
        v = v / np.linalg.norm(v)
    log_growth, M = 0.0, 300
    for _ in range(M):
        v = J @ v
        nrm = np.linalg.norm(v)
        log_growth += np.log(nrm)
        v = v / nrm
    rate = log_growth / M
    assert rate == pytest.approx(parity_mle(A), abs=1e-6)


def test_defect_pattern_matches_gf2_power_and_is_binary():
    A = _adj(nx.watts_strogatz_graph(40, 4, 0.3, seed=5))
    seed, T = 3, 12
    pattern = defect_pattern(A, seed=seed, T=T)          # (T+1, N)
    assert set(np.unique(pattern)).issubset({0, 1})
    for t in range(T + 1):
        col = gf2_matrix_power(A, t)[:, seed]
        assert np.array_equal(pattern[t], col)


def test_ring_defect_is_sierpinski_kummer_count():
    # On a ring, the rule-90 defect count at time t equals the number of odd
    # entries in row t of Pascal's triangle = 2**popcount(t)  (Kummer/Sierpinski).
    N = 401
    A = _adj(nx.cycle_graph(N))
    seed = N // 2
    T = 40
    pattern = defect_pattern(A, seed=seed, T=T, self_inclusive=False)
    for t in [1, 2, 3, 7, 8, 15, 16, 31]:
        assert pattern[t].sum() == 2 ** bin(t).count("1")
