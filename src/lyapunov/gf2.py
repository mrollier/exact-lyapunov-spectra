"""Integer- and GF(2)-safe matrix powers.

This module exists to avoid the numerical trap at the heart of the paper:
``numpy.linalg.matrix_power`` (and any plain ``@`` on ``int64`` arrays)
**overflows silently**. The adjacency-matrix power ``A**t`` counts walks of
length ``t``, and those counts grow exponentially, so for even moderate ``t`` the
int64 accumulator wraps around and returns nonsense without raising anything.

The defect pattern of the parity rule is ``A**t e_j (mod 2)``. There is a
subtlety worth stating precisely: a *pure int64* power wraps modulo ``2**64`` and
``2`` divides ``2**64``, so ``matrix_power(int64) % 2`` happens to keep the
correct parity even after overflow -- only the magnitudes are wrong. The parity
*is* destroyed, however, when the power is formed in floating point (``float64``
loses integer precision above ``2**53``, so the low bit is lost) and then reduced
modulo two -- which is how matrix powers are most often computed. The GF(2)
routine below is correct in every case and never overflows, so all defect-pattern
code uses it rather than relying on that two's-complement coincidence.

Two safe routines are provided:

* :func:`int_matrix_power` -- exact integer power using Python ``int`` objects
  (arbitrary precision), for when the true walk counts are needed.
* :func:`gf2_matrix_power` -- power over the field GF(2), reducing ``mod 2`` at
  every multiplication so the entries never leave ``{0, 1}`` and overflow is
  impossible by construction. This is what the parity-rule defect patterns use.

Never replace these with ``numpy.linalg.matrix_power`` on an integer matrix.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _to_object_pyint(A) -> NDArray:
    """Return a copy of ``A`` as an object array whose entries are Python ``int``.

    Casting an ``int64`` array with ``.astype(object)`` keeps the elements as
    ``numpy.int64`` *scalars*, whose arithmetic still overflows. We therefore
    coerce each entry to a genuine Python ``int``, which has arbitrary precision.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square 2-D matrix, got shape {A.shape}.")
    out = np.empty(A.shape, dtype=object)
    for idx in np.ndindex(A.shape):
        out[idx] = int(A[idx])
    return out


def _identity_object(n: int) -> NDArray:
    """Identity matrix of Python ints (object dtype)."""
    eye = np.zeros((n, n), dtype=object)
    for i in range(n):
        eye[i, i] = 1
    return eye


def int_matrix_power(A, t: int) -> NDArray:
    """Exact integer matrix power ``A**t`` with no overflow.

    Uses binary exponentiation over object-dtype arrays of Python ints, so the
    result is exact regardless of how large the entries become.

    Parameters
    ----------
    A : array_like
        Square integer matrix.
    t : int
        Non-negative exponent.

    Returns
    -------
    numpy.ndarray (dtype=object)
        ``A**t`` with entries as exact Python ints.
    """
    if t < 0:
        raise ValueError(f"Exponent must be non-negative, got t={t}.")
    base = _to_object_pyint(A)
    n = base.shape[0]
    result = _identity_object(n)
    while t:
        if t & 1:
            result = np.dot(result, base)
        t >>= 1
        if t:
            base = np.dot(base, base)
    return result


def gf2_matmul(A, B) -> NDArray:
    """Matrix product over GF(2): ``(A @ B) mod 2``.

    Entries stay in ``{0, 1}``. Because each accumulator is bounded by the shared
    dimension (well below int64 range for any realistic graph), the int64 product
    cannot overflow before the reduction; we still reduce immediately for safety.
    """
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    return np.mod(A @ B, 2)


def gf2_matrix_power(A, t: int) -> NDArray:
    """Matrix power over GF(2): ``A**t mod 2``.

    Binary exponentiation with a ``mod 2`` after every multiplication, so entries
    never leave ``{0, 1}`` and overflow is impossible by construction.

    Parameters
    ----------
    A : array_like
        Square 0/1 matrix (e.g. an adjacency matrix).
    t : int
        Non-negative exponent.

    Returns
    -------
    numpy.ndarray (dtype=int64)
        ``A**t mod 2`` with entries in ``{0, 1}``.
    """
    if t < 0:
        raise ValueError(f"Exponent must be non-negative, got t={t}.")
    base = np.mod(np.asarray(A, dtype=np.int64), 2)
    n = base.shape[0]
    if base.ndim != 2 or base.shape[1] != n:
        raise ValueError(f"Expected a square 2-D matrix, got shape {base.shape}.")
    result = np.eye(n, dtype=np.int64)
    while t:
        if t & 1:
            result = gf2_matmul(result, base)
        t >>= 1
        if t:
            base = gf2_matmul(base, base)
    return result
