"""Configuration-dependent Boolean Jacobians and Benettin along a trajectory.

For an affine ECA the Boolean Jacobian is constant, which is what makes the exact
spectrum of :mod:`lyapunov.spectra` possible. For the other 240 rules the Boolean
derivatives depend on the neighbourhood, so the Jacobian changes at every step and
the tangent map over ``T`` steps is the ordered product ``J_{T-1} ... J_1 J_0``
with ``J_t`` evaluated at the configuration reached at time ``t``. This module
supplies that product and nothing else: the derivatives themselves come from
:func:`lyapunov.rules.gradient_truth_tables`, which is already defined for all
256 rules, and the evolution from :func:`lyapunov.jacobian.eca_step`.

Three points of substance.

* **Banded propagation.** The Jacobian of a radius-one ECA is a circulant-banded
  0/1 matrix, so ``J Q`` costs ``O(N^2)`` rather than the ``O(N^3)`` of a dense
  product. :func:`apply_bands` does this; at ``N = 1000`` it is the difference
  between a feasible and an infeasible run. The dense matrix is still available
  from :func:`eca_jacobian_at` for checking.

* **Collapsed directions, counted exactly.** A non-affine Jacobian is almost
  always singular, so some directions are annihilated and their exponents are
  ``-inf``. How many is a rank question, and it must not be answered by
  thresholding the QR diagonal. For the rules whose singularity comes from a
  vanishing row (6, 73, 41) the corresponding ``|R_ii|`` do fall to 1e-33 and
  below, but for the others (26, 122, 126, 54, 110) they form an unbroken
  continuum from 1e-6 down to 1e-18, with no gap at which to cut: an unpivoted
  QR is not rank-revealing. :func:`tangent_rank` therefore computes the rank of
  the window product in exact arithmetic, and ``N - rank`` is the number of
  exponents that are ``-inf``. The log-stretch array carries ``-inf`` only
  where a pivot is exactly zero, and is not otherwise censored.

* **Vispoel's method along a trajectory.**
  :func:`direct_multiplication_trajectory` is the time-varying counterpart of
  :func:`lyapunov.benettin.direct_multiplication_unscaled`, with the same
  semantics: no rescaling, eigenvalues of ``Y Y^T``, ``nan`` where those are not
  positive, ``OverflowError`` where the product leaves float64.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .jacobian import eca_step
from .rules import gradient_truth_tables

def _check_state(state: NDArray) -> NDArray[np.int_]:
    """Validate a ring configuration and return it as an integer array."""
    state = np.asarray(state, dtype=int)
    if state.ndim != 1:
        raise ValueError(f"The configuration must be one-dimensional, got shape {state.shape}.")
    if state.size < 3:
        raise ValueError(f"Need N >= 3 cells on a ring, got N={state.size}.")
    if np.any((state != 0) & (state != 1)):
        raise ValueError("The configuration must be binary (entries 0 or 1).")
    return state


def eca_gradient_bands(rule: int, state: NDArray) -> NDArray[np.int_]:
    """The three Boolean-derivative bands of ``rule`` at ``state``, shape (3, N).

    Row 0 is ``dphi/ds_{i-1}``, row 1 ``dphi/ds_i`` and row 2 ``dphi/ds_{i+1}``,
    each evaluated cell by cell at the neighbourhood actually present in
    ``state`` (periodic boundary). For an affine rule every column is the same
    and equals the rule's constant gradient.
    """
    state = _check_state(state)
    code = 4 * np.roll(state, 1) + 2 * state + np.roll(state, -1)
    return gradient_truth_tables(rule)[:, code]


def eca_jacobian_at(rule: int, state: NDArray) -> NDArray[np.int_]:
    """Dense Boolean Jacobian of ``rule`` at the configuration ``state``.

    Row ``i`` carries that cell's three derivatives on the sub-diagonal, the
    diagonal and the super-diagonal, with periodic wraparound. For an affine
    rule the result is independent of ``state`` and equals
    :func:`lyapunov.jacobian.eca_jacobian`.
    """
    bands = eca_gradient_bands(rule, state)
    N = bands.shape[1]
    J = np.zeros((N, N), dtype=int)
    i = np.arange(N)
    J[i, (i - 1) % N] = bands[0]
    J[i, i] = bands[1]
    J[i, (i + 1) % N] = bands[2]
    return J


def apply_bands(bands: NDArray, Q: NDArray) -> NDArray[np.floating]:
    """``J @ Q`` for the banded Jacobian described by ``bands``, in O(N^2).

    ``(J Q)[i] = a_-[i] Q[i-1] + a_o[i] Q[i] + a_+[i] Q[i+1]``, with the row
    indices taken modulo ``N``. Identical to ``eca_jacobian_at(...) @ Q`` up to
    the order of three additions, and several times faster because it never
    forms the ``N x N`` matrix whose rows are all but three zeros.
    """
    bands = np.asarray(bands, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[0] != 3:
        raise ValueError(f"bands must have shape (3, N), got {bands.shape}.")
    if Q.ndim != 2 or Q.shape[0] != bands.shape[1]:
        raise ValueError(f"Q must have {bands.shape[1]} rows, got shape {Q.shape}.")
    return (bands[0][:, None] * np.roll(Q, 1, axis=0)
            + bands[1][:, None] * Q
            + bands[2][:, None] * np.roll(Q, -1, axis=0))


def _starting_frame(N: int, Q0: Optional[NDArray]) -> NDArray[np.floating]:
    """The orthonormal frame Benettin starts from: the identity unless given."""
    if Q0 is None:
        return np.eye(N)
    Q = np.array(Q0, dtype=np.float64)
    if Q.shape != (N, N):
        raise ValueError(f"Q0 must be {N} x {N}, got {Q.shape}.")
    return Q


def benettin_log_stretch_trajectory(
    rule: int,
    state0: NDArray,
    T: int,
    Q0: Optional[NDArray] = None,
) -> NDArray[np.floating]:
    """Per-step log stretching factors along a trajectory, shape (T, N).

    The configuration starts at ``state0`` and is advanced by
    :func:`lyapunov.jacobian.eca_step`; at step ``t`` the frame is propagated by
    the Jacobian evaluated at the configuration present at that step, then
    re-orthonormalised. ``L[t, i]`` is ``ln|R_ii|``, which is ``-inf`` exactly
    when the pivot is exactly zero. It is deliberately not thresholded: for a
    singular Jacobian the annihilated direction usually shows up as a pivot of
    order the unit round-off rather than as a hard zero, and there is no gap in
    the distribution at which a cut-off could be placed. The number of ``-inf``
    exponents is :func:`tangent_rank`'s business, not this function's.

    This is the time-varying counterpart of
    :func:`lyapunov.benettin.benettin_log_stretch`, and on an affine rule it
    reproduces it exactly. As there, every finite-time estimator is an offline
    slice of the result, so one run serves every burn-in and window length: pass
    the array to :func:`lyapunov.benettin.windowed_spectrum` or
    :func:`lyapunov.benettin.cumulative_spectrum`.

    Row sums equal ``ln|det J_t|`` at every step, which is the C6 identity.
    Where ``J_t`` is singular that value is ``-inf`` and the computed sum is
    instead large and negative, of order ``ln(eps)``, for the reason just given.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    state = _check_state(state0)
    N = state.size
    Q = _starting_frame(N, Q0)
    L = np.empty((T, N))
    for t in range(T):
        bands = eca_gradient_bands(rule, state)
        Q, R = np.linalg.qr(apply_bands(bands, Q))
        diag = np.diag(R)
        # Fix the sign ambiguity of QR so the stretching factors are positive.
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        Q = Q * signs
        with np.errstate(divide="ignore"):
            L[t] = np.log(np.abs(diag))     # -inf only for an exactly zero pivot
        state = eca_step(rule, state)
    return L


def spectrum_along_trajectory(
    rule: int,
    state0: NDArray,
    T: int,
    burn: int = 0,
    Q0: Optional[NDArray] = None,
) -> NDArray[np.floating]:
    """Finite-time spectrum over steps ``burn + 1 .. T``, sorted descending.

    A convenience wrapper: one QR run of ``T`` steps, averaged over the window
    that follows the burn-in by :func:`lyapunov.benettin.windowed_spectrum`.

    The bottom ``N - tangent_rank(rule, state0, burn, T - burn)`` entries are the
    exponents that are truly ``-inf``; some of them come back as large finite
    negative numbers, because the QR pivots of an annihilated direction are of
    order the unit round-off rather than zero. Censor by the rank, not by the
    value.
    """
    from .benettin import windowed_spectrum  # local: benettin does not import this module

    if not 0 <= burn < T:
        raise ValueError(f"Need 0 <= burn < T, got burn={burn}, T={T}.")
    L = benettin_log_stretch_trajectory(rule, state0, T, Q0=Q0)
    return windowed_spectrum(L, burn, T - burn)


def direct_multiplication_trajectory(rule: int, state0: NDArray, T: int) -> NDArray[np.floating]:
    """Vispoel et al. (2024) for a configuration-dependent Jacobian.

    Their Eqs. (8)-(10) and (35) with the constant ``J`` replaced by the
    Jacobian at each step, which is what a non-affine rule requires: ``Y^0 = I``,
    ``Y^{t+1} = J_t Y^t`` with no rescaling, then
    ``Lambda_k = ln(mu_k) / (2T)`` for the eigenvalues ``mu_k`` of ``Y (Y)^T``
    in descending order, everything in float64.

    The semantics are those of
    :func:`lyapunov.benettin.direct_multiplication_unscaled`: eigenvalues at or
    below zero are rounding noise and are returned as ``nan`` rather than
    clamped, and an ``OverflowError`` is raised when the unscaled product leaves
    float64 instead of returning ``inf``. The resolvable range is bounded below
    by :func:`lyapunov.benettin.precision_floor` evaluated at the rule's own
    ``sigma_max = exp(MLE)``, which for a horizon of 500 steps sits only
    ``|ln(eps)| / 1000 = 0.036`` below the maximal exponent.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    state = _check_state(state0)
    N = state.size
    Y = np.eye(N)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(T):
            Y = apply_bands(eca_gradient_bands(rule, state), Y)
            state = eca_step(rule, state)
        G = Y @ Y.T
    overflow = OverflowError(
        f"Y Y^T overflows float64 for rule {rule} at T = {T}: its largest "
        "eigenvalue is of order exp(2 T * MLE), so the unscaled method has no "
        "result once that exceeds about 1.8e308."
    )
    if not np.all(np.isfinite(G)):
        raise overflow
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    if not np.all(np.isfinite(ev)):
        raise overflow
    out = np.full(N, np.nan)
    positive = ev > 0
    out[positive] = np.log(ev[positive]) / (2 * T)
    return out


# Primes for the exact rank. The window product is accumulated with
# float64 arithmetic and reduced after every step; each output entry is a sum of
# three products of residues, so ``3 * p**2`` must stay below 2**53 for the
# accumulation to be exact. Both primes satisfy that with room to spare.
RANK_MODULUS = 46337
RANK_MODULUS_ALT = 40009


def integer_matrix_rank(A: NDArray, p: int = None) -> int:
    """Exact rank of an integer matrix, by Gaussian elimination over GF(p).

    The rank over GF(p) can only fall short of the rank over the rationals, and
    does so only if ``p`` divides every minor of that order; :data:`RANK_MODULUS`
    and :data:`RANK_MODULUS_ALT` are provided so the answer can be cross-checked.
    Entries must already be reducible mod ``p`` without loss, which they are for
    the 0/1 Jacobians and for anything :func:`window_product_mod_p` returns.
    """
    if p is None:
        p = RANK_MODULUS
    A = np.asarray(A, dtype=np.int64) % p
    n, m = A.shape
    r = 0
    for c in range(m):
        nonzero = np.nonzero(A[r:, c])[0]
        if nonzero.size == 0:
            continue
        i = r + int(nonzero[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p   # Fermat inverse
        below = np.nonzero(A[r + 1:, c])[0] + r + 1
        if below.size:
            A[below] = (A[below] - np.outer(A[below, c], A[r])) % p
        r += 1
        if r == n:
            break
    return r


def window_product_mod_p(
    rule: int, state0: NDArray, burn: int, window: int, p: int = RANK_MODULUS
) -> NDArray[np.int_]:
    """The tangent map over the window, reduced mod ``p``: exact, never overflows.

    The Jacobians are 0/1 integer matrices, so their ordered product is an
    integer matrix. Forming it in floating point would overflow long before the
    window is out (its entries grow like ``exp(T * MLE)``), but reducing mod a
    prime after every step keeps every entry below ``p`` while leaving the
    result exact.
    """
    if burn < 0 or window < 1:
        raise ValueError(f"Need burn >= 0 and window >= 1, got burn={burn}, window={window}.")
    state = _check_state(state0)
    for _ in range(burn):
        state = eca_step(rule, state)
    P = np.eye(state.size)
    for _ in range(window):
        P = apply_bands(eca_gradient_bands(rule, state), P) % p
        state = eca_step(rule, state)
    return P.astype(np.int64)


def tangent_rank(
    rule: int, state0: NDArray, burn: int, window: int, p: int = RANK_MODULUS
) -> int:
    """Exact rank of the tangent map over steps ``burn + 1 .. burn + window``.

    ``N - tangent_rank(...)`` is the number of Lyapunov exponents that are
    exactly ``-inf`` over that window: those directions are annihilated
    algebraically, not merely shrunk. This is the honest count, and it is not
    the same as the number of zero pivots Benettin's QR happens to produce,
    because an unpivoted QR is not rank-revealing: for rules 26 and 122 the
    floating-point iteration returns a finite (very negative) exponent for
    directions that are in fact annihilated.

    The rank is taken over GF(``p``), which can only underestimate the rank over
    the rationals, and does so only if ``p`` divides every minor of the relevant
    order. Two different primes are provided (:data:`RANK_MODULUS` and
    :data:`RANK_MODULUS_ALT`) so the result can be cross-checked; the
    verification suite does that.
    """
    return integer_matrix_rank(window_product_mod_p(rule, state0, burn, window, p), p)
