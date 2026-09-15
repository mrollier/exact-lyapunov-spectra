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

Five further routines support the convergence study of
``notebooks/03_benettin_convergence.ipynb`` and the revised Figure 3 of
``figures/make_convergence_figure.py``, which use the exact affine spectrum
to calibrate the two numerical routes:

* :func:`benettin_running_average` -- Benettin with an optional burn-in, an
  optional starting frame and running averages read off at several horizons in
  one pass.
* :func:`direct_multiplication_unscaled` -- the direct method exactly as stated
  by Vispoel et al. (2024), Eqs. (8)-(10) and (35): the unscaled product
  ``Y^T = J^T`` and the eigenvalues of ``Y Y^T``, in float64. Unlike
  :func:`direct_multiplication_spectrum` it does *not* rescale, so it overflows
  float64 once ``sigma_max^(2T)`` exceeds about ``1.8e308``.
* :func:`precision_floor` -- the exponent below which the unscaled method can
  resolve nothing in double precision.
* :func:`closed_form_spectrum` -- the exact affine spectrum ``ln(sigma_k)``,
  sorted descending with ``-inf`` for the zero singular values: the reference
  every estimator is compared against.
* :func:`benettin_log_stretch` -- one QR run, storing the per-step log
  stretching factors so that every finite-time estimator (running average,
  burn-in plus window) is an offline slice: :func:`windowed_spectrum`,
  :func:`cumulative_spectrum` (the windowed convergence figure).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

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


def benettin_running_average(
    J: NDArray,
    T: int,
    burn: int = 0,
    checkpoints: Optional[Iterable[int]] = None,
    Q0: Optional[NDArray] = None,
) -> Dict[int, NDArray[np.floating]]:
    """Benettin running averages at several horizons in a single pass.

    Same iteration as :func:`benettin_spectrum` (``Z = J Q``, ``Z = Q R``,
    accumulate ``ln|R_ii|``), with three additions used by the convergence
    study:

    * ``burn`` QR steps are taken *before* accumulation starts. They align the
      frame with the singular directions and contribute nothing to the average.
    * ``Q0`` is the starting orthonormal frame (the identity if ``None``). From
      the eigenbasis of a normal ``J`` the factor ``R`` is diagonal at every
      step and the average is exact from ``T = 1``.
    * ``checkpoints`` are the horizons at which the running average ``acc / t``
      is recorded; the result maps each checkpoint to that average. If
      ``None``, only ``T`` is recorded.

    The averages are returned in *frame order* (``k``-th diagonal entry of
    ``R``), not sorted: the ``k``-th direction of Benettin's frame converges to
    the ``k``-th largest exponent, and the study measures exactly that
    convergence. Sorting would hide it.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    if burn < 0:
        raise ValueError(f"Need burn >= 0, got {burn}.")
    J = np.asarray(J, dtype=np.float64)
    N = J.shape[0]
    Q = np.eye(N) if Q0 is None else np.array(Q0, dtype=np.float64, copy=True)
    if Q.shape != (N, N):
        raise ValueError(f"Q0 must be {N} x {N}, got {Q.shape}.")
    checkpoints = {T} if checkpoints is None else set(int(c) for c in checkpoints)
    if any(c < 1 or c > T for c in checkpoints):
        raise ValueError(f"Checkpoints must lie in 1..{T}, got {sorted(checkpoints)}.")

    def step(Q):
        Q, R = np.linalg.qr(J @ Q)
        diag = np.diag(R)
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        return Q * signs, np.abs(diag)

    for _ in range(burn):
        Q, _ = step(Q)
    acc = np.zeros(N)
    out: Dict[int, NDArray[np.floating]] = {}
    for t in range(1, T + 1):
        Q, stretch = step(Q)
        acc += np.log(stretch)
        if t in checkpoints:
            out[t] = acc / t
    return out


def direct_multiplication_unscaled(J: NDArray, T: int) -> NDArray[np.floating]:
    """Direct multiplication exactly as stated by Vispoel et al. (2024).

    Their Eqs. (8)-(10) and (35): ``Y^0 = I``, ``Y^{t+1} = J Y^t`` with no
    rescaling, then ``Lambda_k = ln(mu_k) / (2T)`` where ``mu_k`` are the
    eigenvalues of ``Y^T (Y^T)^T`` in descending order. Everything is float64.

    Two consequences, both demonstrated in the convergence notebook:

    * The eigenvalues of ``Y Y^T`` span ``sigma_max^(2T)`` down to
      ``sigma_min^(2T)``; double precision resolves them only down to
      ``eps * sigma_max^(2T)``. Every eigenvalue below that is rounding
      noise, spread on both sides of zero: the positive ones read as the
      :func:`precision_floor`, the negative ones (typically a third to a half
      of the spectrum) have no logarithm and are returned as ``nan``. Nothing
      is clamped; a ``nan`` means the method produced no number for that
      exponent.
    * The entries of ``Y`` grow like ``sigma_max^T``; once ``Y Y^T`` exceeds
      the largest float64 the method has no result at all. An ``OverflowError``
      is raised rather than returning ``inf``/``nan``.

    This is deliberately *not* :func:`direct_multiplication_spectrum`, which
    rescales every step to stay finite. No ``numpy.linalg.matrix_power`` is
    used; the product is formed step by step, as in the cited equations.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    J = np.asarray(J, dtype=np.float64)
    N = J.shape[0]
    Y = np.eye(N)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(T):
            Y = J @ Y
        G = Y @ Y.T
    ev = np.full(N, np.nan)
    if np.all(np.isfinite(G)):
        ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    if not np.all(np.isfinite(ev)):
        # The largest eigenvalue of Y Y^T is sigma_max^(2T); it exceeds the
        # largest float64 one step before the entries of Y Y^T themselves do.
        raise OverflowError(
            f"Y Y^T overflows float64 at T = {T}: its largest eigenvalue is "
            "sigma_max^(2T), so the unscaled method has no result beyond "
            "T = floor(ln(1.8e308) / (2 ln sigma_max))."
        )
    out = np.full(N, np.nan)
    positive = ev > 0
    out[positive] = np.log(ev[positive]) / (2 * T)
    return out


def precision_floor(sigma_max: float, T: int, eps: float = float(np.finfo(np.float64).eps)) -> float:
    """Lowest exponent the unscaled direct method can resolve in float64.

    The eigenvalues of ``Y Y^T`` are ``sigma_k^(2T)``. In floating point with
    unit round-off ``eps`` they all carry an absolute error of about
    ``eps * sigma_max^(2T)``, so any ``sigma_k^(2T)`` below that is
    indistinguishable from rounding noise. Taking ``ln(.) / (2T)`` gives

        floor(T) = ln(sigma_max) + ln(eps) / (2T),

    which *rises* towards the MLE as ``T`` grows: more timesteps make the method
    worse, not better. Every true exponent below the floor is reported at (or
    near) the floor.
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    return float(np.log(sigma_max) + np.log(eps) / (2 * T))


def closed_form_spectrum(rule: int, N: int) -> NDArray[np.floating]:
    """Exact affine-ECA Lyapunov spectrum ``ln(sigma_k)``, sorted descending.

    Zero singular values map to ``-inf`` and are placed last.
    """
    sv = eca_singular_values(rule, N)
    with np.errstate(divide="ignore"):
        lyap = np.log(sv)
    return np.sort(lyap)[::-1]


def benettin_log_stretch(J: NDArray, T: int, Q0: Optional[NDArray] = None) -> NDArray[np.floating]:
    """Per-step log stretching factors of Benettin's QR iteration, shape (T, N).

    Runs ``T`` steps of ``Z = J Q``, ``Q, R = qr(Z)`` from the orthonormal frame
    ``Q0`` (identity by default) and stores ``ln|R_ii|`` at every step, column
    ``i`` being the ``i``-th Gram-Schmidt direction. Every finite-time
    Lyapunov estimator is then a mean over a slice of rows:
    :func:`cumulative_spectrum` for the running average from step one and
    :func:`windowed_spectrum` for a window taken after a burn-in. One run
    therefore serves every burn-in and window length. Row sums equal
    ``ln|det J|`` at every step (C6).
    """
    if T < 1:
        raise ValueError(f"Need T >= 1, got {T}.")
    J = np.asarray(J, dtype=np.float64)
    N = J.shape[0]
    if Q0 is None:
        Q = np.eye(N)
    else:
        Q = np.array(Q0, dtype=np.float64)
        if Q.shape != (N, N):
            raise ValueError(f"Q0 must be {N} x {N}, got {Q.shape}.")
    L = np.empty((T, N))
    for t in range(T):
        Q, R = np.linalg.qr(J @ Q)
        diag = np.diag(R)
        signs = np.sign(diag)
        signs[signs == 0] = 1.0
        Q = Q * signs
        L[t] = np.log(np.abs(diag))
    return L


def windowed_spectrum(L: NDArray, burn: int, window: int) -> NDArray[np.floating]:
    """Finite-time exponents over steps ``burn + 1 .. burn + window``.

    ``L`` is the array from :func:`benettin_log_stretch`. Returns the mean of
    those rows, sorted in descending order. Sorting is essential: nearly
    degenerate pairs exchange places while converging, so a per-column
    comparison with the sorted exact spectrum would be wrong.
    """
    L = np.asarray(L, dtype=np.float64)
    if burn < 0 or window < 1:
        raise ValueError(f"Need burn >= 0 and window >= 1, got burn={burn}, window={window}.")
    if burn + window > L.shape[0]:
        raise ValueError(f"burn + window = {burn + window} exceeds the {L.shape[0]} stored steps.")
    return np.sort(L[burn:burn + window].mean(axis=0))[::-1]


def cumulative_spectrum(L: NDArray, T: int) -> NDArray[np.floating]:
    """The running average from step one over ``T`` steps (the estimator of
    :func:`benettin_spectrum`), sorted in descending order: a window of length
    ``T`` with no burn-in, so it carries the initial transient diluted as 1/T.
    """
    return windowed_spectrum(L, 0, T)
