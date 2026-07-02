"""Constant Boolean Jacobians and pure-numpy ECA evolution.

For an affine ECA on a periodic ring the Boolean Jacobian is a circulant matrix
whose every row carries the gradient ``(a_-, a_o, a_+)`` shifted by one cell
(Eq. 7 of the paper). For the parity rule on a lattice the Jacobian is the
lattice adjacency matrix (optionally plus the identity for the self-inclusive
case). Both are built here explicitly so the closed-form spectra can be
cross-checked against a direct singular-value decomposition.

The ECA evolution helpers use plain numpy with periodic boundary conditions and
avoid any external CA library.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .rules import affine_gradient, truth_table


def eca_jacobian(rule: int, N: int) -> NDArray[np.int_]:
    """Constant circulant Jacobian of an affine ECA on a ring of ``N`` cells.

    Row ``i`` has ``a_o`` on the diagonal, ``a_+`` on the super-diagonal and
    ``a_-`` on the sub-diagonal (all with periodic wraparound), where
    ``(a_-, a_o, a_+)`` is the rule's gradient.

    Raises ``ValueError`` if the rule is not affine (then the Jacobian is not
    constant and this construction does not apply).
    """
    a_minus, a_centre, a_plus = affine_gradient(rule)  # raises if not affine
    if N < 3:
        raise ValueError(f"Need N >= 3 for a ring ECA Jacobian, got N={N}.")
    J = np.zeros((N, N), dtype=int)
    for i in range(N):
        J[i, i] = a_centre
        J[i, (i + 1) % N] = a_plus
        J[i, (i - 1) % N] = a_minus
    return J


def eca_step(rule: int, state: NDArray[np.int_]) -> NDArray[np.int_]:
    """One synchronous ECA step on a ring (periodic boundary), vectorised."""
    state = np.asarray(state, dtype=int)
    left = np.roll(state, 1)    # neighbour s_{i-1}
    right = np.roll(state, -1)  # neighbour s_{i+1}
    code = 4 * left + 2 * state + right
    table = truth_table(rule)
    return table[code]


def eca_evolve(rule: int, state0: NDArray[np.int_], T: int) -> NDArray[np.int_]:
    """Evolve an ECA for ``T`` steps; return an array of shape ``(T + 1, N)``."""
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}.")
    N = len(state0)
    out = np.empty((T + 1, N), dtype=int)
    out[0] = np.asarray(state0, dtype=int)
    for t in range(1, T + 1):
        out[t] = eca_step(rule, out[t - 1])
    return out


def _torus_index(coords: Sequence[int], L: int) -> int:
    """Lexicographic flat index of lattice ``coords`` on an ``L``-per-side torus."""
    idx = 0
    for c in coords:
        idx = idx * L + (c % L)
    return idx


def build_torus_parity_jacobian(
    offsets: Sequence[Tuple[int, ...]], L: int
) -> NDArray[np.int_]:
    """Constant Jacobian of the parity rule on a ``D``-dimensional torus.

    ``offsets`` lists the neighbour displacement vectors (including ``(0, ..., 0)``
    for the self-inclusive parity rule). The dimension ``D`` is inferred from the
    length of an offset vector; the lattice has ``L`` cells per side and
    ``N = L**D`` nodes. The result is the 0/1 matrix ``J`` with ``J[i, j] = 1``
    iff node ``j`` is at one of the offsets from node ``i``.
    """
    offsets = [tuple(o) for o in offsets]
    D = len(offsets[0])
    if any(len(o) != D for o in offsets):
        raise ValueError("All offset vectors must have the same dimension.")
    N = L ** D
    J = np.zeros((N, N), dtype=int)
    # iterate over every lattice site by its D-dimensional coordinates
    for flat in range(N):
        coords = []
        rem = flat
        for _ in range(D):
            coords.append(rem % L)
            rem //= L
        coords = coords[::-1]  # match lexicographic order used in _torus_index
        i = _torus_index(coords, L)
        for off in offsets:
            neigh = [coords[d] + off[d] for d in range(D)]
            j = _torus_index(neigh, L)
            J[i, j] = 1
    return J
