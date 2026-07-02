"""The parity rule on an arbitrary graph.

For the parity rule ``s_i <- (a_o s_i) XOR (XOR of neighbours)`` the Boolean
Jacobian is exactly the adjacency matrix (plus the identity when the rule is
self-inclusive, ``a_o = 1``). Because ``A`` is real and symmetric, the whole
Lyapunov analysis reduces to the graph's eigen-structure:

* Lyapunov spectrum  = ``ln|lambda_k(A) + a_o|``     (the absolute graph spectrum);
* MLE                = ``ln(rho(A) + a_o)``          (log of the spectral radius);
* long-time single-site amplitude  proportional to the seeded node's eigenvector
  centrality (the principal eigenvector component);
* true defect pattern = ``A^t e_j (mod 2)``          (walk-counting modulo two).

All matrix powers used for defect patterns go through :mod:`lyapunov.gf2` so the
walk counts never overflow before the ``mod 2`` reduction.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .gf2 import gf2_matmul


def _check_adjacency(A: NDArray) -> NDArray:
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {A.shape}.")
    return A


def parity_jacobian(A: NDArray, self_inclusive: bool = False) -> NDArray[np.int_]:
    """Constant Jacobian of the parity rule: ``A`` (+ identity if self-inclusive)."""
    A = _check_adjacency(A).astype(int)
    if self_inclusive:
        A = A + np.eye(A.shape[0], dtype=int)
    return A


def parity_lyapunov_spectrum(
    A: NDArray, self_inclusive: bool = False, drop_zeros: bool = True
) -> NDArray[np.floating]:
    """Lyapunov spectrum ``ln|lambda_k(A) + a_o|`` of the parity rule."""
    A = _check_adjacency(A).astype(float)
    a_o = 1.0 if self_inclusive else 0.0
    eig = np.linalg.eigvalsh(A) + a_o
    sv = np.abs(eig)
    if drop_zeros:
        sv = sv[sv > 0]
    with np.errstate(divide="ignore"):
        return np.log(sv)


def parity_mle(A: NDArray, self_inclusive: bool = False) -> float:
    """MLE of the parity rule: ``ln(rho(A) + a_o)`` (log of the spectral radius)."""
    A = _check_adjacency(A).astype(float)
    a_o = 1.0 if self_inclusive else 0.0
    rho = np.max(np.abs(np.linalg.eigvalsh(A) + a_o))
    with np.errstate(divide="ignore"):
        return float(np.log(rho))


def eigenvector_centrality(A: NDArray) -> NDArray[np.floating]:
    """Principal (Perron) eigenvector of ``A``, unit-norm and sign-fixed positive.

    On an undirected connected graph this is the eigenvector centrality.
    """
    A = _check_adjacency(A).astype(float)
    w, V = np.linalg.eigh(A)
    x = V[:, -1]  # eigenvector of the largest eigenvalue
    if np.sum(x) < 0:
        x = -x
    return x / np.linalg.norm(x)


def long_time_amplitude_ratio(
    A: NDArray, T: int, self_inclusive: bool = False
) -> NDArray[np.floating]:
    """Per-node ``||J^T e_i|| / lambda_N^T`` -- the long-time perturbation amplitude.

    Computed from the eigendecomposition (so it never overflows), this converges
    as ``T -> infinity`` to the eigenvector centrality of node ``i``.
    """
    A = _check_adjacency(A).astype(float)
    a_o = 1.0 if self_inclusive else 0.0
    w, V = np.linalg.eigh(A)
    lam = w + a_o
    lam_N = lam[np.argmax(np.abs(lam))]
    ratio_sq = np.sum((V ** 2) * (lam / lam_N) ** (2 * T), axis=1)
    return np.sqrt(ratio_sq)


def defect_pattern(
    A: NDArray, seed: int, T: int, self_inclusive: bool = False
) -> NDArray[np.int_]:
    """True Boolean defect pattern ``Delta s^t = A^t e_seed (mod 2)`` for t=0..T.

    Returns an array of shape ``(T + 1, N)`` of 0/1 values. The iteration applies
    the Jacobian modulo two at every step, so it is exact and cannot overflow.
    """
    J = parity_jacobian(A, self_inclusive)
    N = J.shape[0]
    if not (0 <= seed < N):
        raise ValueError(f"seed must be in 0..{N - 1}, got {seed}.")
    out = np.zeros((T + 1, N), dtype=int)
    v = np.zeros(N, dtype=int)
    v[seed] = 1
    out[0] = v
    for t in range(1, T + 1):
        v = gf2_matmul(J, v.reshape(-1, 1)).ravel()
        out[t] = v
    return out
