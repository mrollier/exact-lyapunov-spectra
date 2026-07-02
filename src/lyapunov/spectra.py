"""Closed-form Lyapunov spectra of affine cellular automata.

For an affine rule the Jacobian is (multilevel) circulant, so it is diagonalised
by the discrete Fourier transform and its singular values are the modulus of the
DFT of the gradient stencil. No trajectory is simulated and no limit is taken:
the Lyapunov spectrum is exactly the logarithm of these singular values.

* 1-D (ECAs):        Eq. (8)-(9) of the paper.
* D-dimensional:     Eq. (12): sigma_k = |sum_delta a_delta exp(2 pi i k.delta / N)|.
* Neighbourhood      Eq. (14): the structure factor K(k, l), the DFT of the
  structure factor:  neighbourhood's indicator function.

The maximal Lyapunov exponent is always ``ln(number of sensitive inputs)``,
attained where every exponential is in phase (Eq. 11/13).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .rules import affine_gradient, gradient_weight

# ---------------------------------------------------------------------------
# 2-D neighbourhoods (offset vectors including the centre, i.e. self-inclusive).
# Removing (0, 0) gives the self-exclusive (genuinely outer-totalistic) case.
# ---------------------------------------------------------------------------
VON_NEUMANN_2D: Tuple[Tuple[int, int], ...] = (
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
)
MOORE_2D: Tuple[Tuple[int, int], ...] = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
)
VON_NEUMANN_R2_2D: Tuple[Tuple[int, int], ...] = tuple(
    (dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 2
)


# ---------------------------------------------------------------------------
# 1-D affine ECAs
# ---------------------------------------------------------------------------
def eca_eigenvalues(rule: int, N: int) -> NDArray[np.complexfloating]:
    """Complex eigenvalues of the affine ECA's circulant Jacobian (Eq. 7)."""
    a_minus, a_centre, a_plus = affine_gradient(rule)
    k = np.arange(N)
    phase = 2j * np.pi * k / N
    return a_centre + a_plus * np.exp(phase) + a_minus * np.exp(-phase)


def eca_singular_values(rule: int, N: int) -> NDArray[np.floating]:
    """Singular values of the affine ECA's constant Jacobian (Eq. 9)."""
    return np.abs(eca_eigenvalues(rule, N))


def eca_lyapunov_spectrum(
    rule: int, N: int, drop_zeros: bool = True
) -> NDArray[np.floating]:
    """Lyapunov spectrum ``ln(sigma_k)`` of an affine ECA (Eq. 10).

    With ``drop_zeros`` (default) the ``-inf`` entries from zero singular values
    are removed, matching how the spectrum is plotted.
    """
    sv = eca_singular_values(rule, N)
    if drop_zeros:
        sv = sv[sv > 0]
    with np.errstate(divide="ignore"):
        return np.log(sv)


def eca_mle(rule: int) -> float:
    """Maximal Lyapunov exponent of an affine ECA: ``ln(gradient weight)``.

    Returns ``-inf`` for the zero-gradient rules (0 and 255), which have no
    growing direction.
    """
    w = gradient_weight(rule)
    return float(np.log(w)) if w > 0 else -np.inf


# ---------------------------------------------------------------------------
# 2-D parity rule (general D-dimensional DFT closed form)
# ---------------------------------------------------------------------------
def parity_2d_singular_values(
    offsets: Sequence[Tuple[int, int]], N: int
) -> NDArray[np.floating]:
    """Singular values of the 2-D parity rule on an ``N x N`` torus (Eq. 12).

    ``offsets`` are the neighbour displacements (include ``(0, 0)`` for the
    self-inclusive rule). Returns the ``N x N`` grid ``sigma_{k,l}``.
    """
    k = np.arange(N)
    kk, ll = np.meshgrid(k, k, indexing="ij")
    total = np.zeros((N, N), dtype=complex)
    for dx, dy in offsets:
        total += np.exp(2j * np.pi * (dx * kk + dy * ll) / N)
    return np.abs(total)


def parity_2d_lyapunov_spectrum(
    offsets: Sequence[Tuple[int, int]], N: int, drop_zeros: bool = True
) -> NDArray[np.floating]:
    """Flattened Lyapunov spectrum ``ln(sigma_{k,l})`` of the 2-D parity rule."""
    sv = parity_2d_singular_values(offsets, N).ravel()
    if drop_zeros:
        sv = sv[sv > 0]
    with np.errstate(divide="ignore"):
        return np.log(sv)


def parity_2d_mle(offsets: Sequence[Tuple[int, int]]) -> float:
    """MLE of the 2-D parity rule: ``ln(number of neighbours)`` (Eq. 13)."""
    return float(np.log(len(offsets)))


def structure_factor(
    offsets: Sequence[Tuple[int, int]], k: int, l: int, N: int
) -> float:
    """Neighbourhood structure factor ``K(k, l)`` (Eq. 14).

    The DFT of the neighbourhood's indicator function, summed over the non-centre
    offsets. Because the neighbourhoods are symmetric under ``delta -> -delta``
    the exponentials pair into cosines, so ``K`` is real.
    """
    a, b = 2 * np.pi * k / N, 2 * np.pi * l / N
    return float(
        sum(
            np.cos(dx * a + dy * b)
            for (dx, dy) in offsets
            if (dx, dy) != (0, 0)
        )
    )
