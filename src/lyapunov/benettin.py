"""Reference numerical Lyapunov routines for the benchmark figure.

Two families of method are provided, to be compared against the exact closed form
of :mod:`lyapunov.spectra`:

* :func:`benettin_spectrum` -- Benettin's algorithm, which interleaves the
  Jacobian product with QR re-orthonormalisations so the slow-growing directions
  are not swamped by the fast ones. Numerically stable, but slow to converge at
  the top of the spectrum.
* :func:`direct_multiplication_spectrum` -- form the Jacobian product directly at
  a chosen floating-point precision, then read the singular values. Exact at the
  top for a normal Jacobian, but at low precision the small singular values are
  lost to rounding. A scalar rescaling at each step keeps the product finite (an
  unscaled product would simply overflow), without otherwise changing the method.

For a constant (affine) Jacobian the product collapses to a matrix power, so both
routines take the single constant matrix ``J`` and a horizon ``T``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spectra import eca_singular_values


def benettin_spectrum(J: NDArray, T: int) -> NDArray[np.floating]:
    """Lyapunov spectrum of a constant Jacobian via Benettin's QR algorithm.

    Starts from an orthonormal frame ``Q = I`` and, at each of ``T`` steps,
    forms ``Z = J Q``, factorises ``Z = Q R`` and accumulates ``ln|R_ii|``. The
    exponents are the time-averaged logarithms of the diagonal stretching
    factors, returned sorted in descending order.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    J = np.asarray(J, dtype=np.float64)
    N = J.shape[0]
    Q = np.eye(N)
    log_stretch = np.zeros(N)
    for _ in range(T):
        Z = J @ Q
        Q, R = np.linalg.qr(Z)
        diag = np.diag(R)
        # Fix the sign ambiguity of QR so the stretching factors are positive.
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        Q = Q * signs
        log_stretch += np.log(np.abs(diag))
    return np.sort(log_stretch / T)[::-1]


def direct_multiplication_spectrum(
    J: NDArray, T: int, dtype=np.float64
) -> NDArray[np.floating]:
    """Lyapunov spectrum of a constant Jacobian by direct multiplication.

    Forms ``J**T`` by repeated multiplication carried out in ``dtype`` (e.g.
    ``np.float16``/``np.float32``/``np.float64``), rescaling by a scalar at each
    step to avoid overflow, then returns ``ln(sigma_i(J**T)) / T`` sorted
    descending. The scalar bookkeeping is done in float64; only the matrix
    product feels the reduced precision, so the characteristic low-precision
    artefact (loss of the small singular values) is reproduced faithfully.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    N = J.shape[0]
    M = np.eye(N, dtype=dtype)
    Jd = np.asarray(J, dtype=dtype)
    log_scale = 0.0  # accumulated ln of the scalar rescalings (float64)
    for _ in range(T):
        M = Jd @ M
        s = np.max(np.abs(M.astype(np.float64)))
        if s == 0:
            # product collapsed to zero; all remaining singular values are zero
            return np.full(N, -np.inf)
        M = (M.astype(np.float64) / s).astype(dtype)
        log_scale += np.log(s)
    sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    with np.errstate(divide="ignore"):
        lyap = (np.log(sv) + log_scale) / T
    return np.sort(lyap)[::-1]


def closed_form_spectrum(rule: int, N: int) -> NDArray[np.floating]:
    """Exact affine-ECA Lyapunov spectrum ``ln(sigma_k)``, sorted descending.

    Zero singular values map to ``-inf`` and are placed last.
    """
    sv = eca_singular_values(rule, N)
    with np.errstate(divide="ignore"):
        lyap = np.log(sv)
    return np.sort(lyap)[::-1]
