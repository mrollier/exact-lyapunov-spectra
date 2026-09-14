"""Boolean damage from a single flipped cell, and the single-vector maximal exponent.

Two quantities are put side by side here, on the same trajectory: the
configuration-space *damage* caused by flipping one cell (the Hamming distance
between the perturbed and the unperturbed evolution) and the tangent-space
*maximal Lyapunov exponent* of the Boolean Jacobian. The first is what the
difference patterns of Fig. 1 show; the second is what the closed form of
:mod:`lyapunov.spectra` and the Benettin runs of :mod:`lyapunov.nonaffine`
compute. Nothing in this module depends on the dimension: the step, the
Jacobian bands and their application are passed in as callables, and the
convenience wrappers at the bottom bind them for a ring ECA and for an
outer-totalistic rule on the torus.

**Damage and its cones.** With one radius-one neighbourhood per step the
damage at time ``t`` lies inside the light cone of the flipped cell: ``2t + 1``
cells on a ring, the L1 ball of ``2t**2 + 2t + 1`` cells on the von Neumann
torus. The lattice is required to hold the whole cone plus an undamaged margin
(``L >= 2T + 3``), so the damage never meets itself round the torus and the
measured quantities do not depend on the lattice size at all. Three views are
recorded, each averaged over the final ``window`` steps:

* ``v_front = r(t) / t``, how far the damage reaches (``r`` is the largest
  distance of a damaged cell from the flip);
* ``D_norm = h(t) / |maximal cone|``, how much damage there is in total;
* ``fill = h(t) / |observed cone of radius r(t)|``, how densely the reached
  region is damaged (0 when the damage has healed).

**Single-vector exponent.** The maximal exponent is the growth rate of one
generic tangent vector under ``J_{T-1} ... J_0``, renormalised after every
step: Benettin's method with a single vector, ``O(N)`` per step through the
banded products. It is the top entry of the full spectrum and nothing else,
which is all that is needed here and the only thing affordable on a torus
with ``L**2`` tangent dimensions. When a rule annihilates the vector exactly
(rules 0, 8, 32, ... on the ring) the exponent is ``-inf`` and the step at
which it died is reported; no threshold is involved, the norm is exactly zero.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .jacobian import eca_step
from .nonaffine import apply_bands, eca_gradient_bands
from .outer_totalistic import (
    MOORE, VON_NEUMANN, Neighbourhood, apply_bands_2d, ot_gradient_bands, ot_step)

Index = Union[int, Sequence[int]]


def maximal_cone(t, dim: int, moore: bool = False):
    """Cells inside the light cone at time ``t``.

    ``2t + 1`` on the ring, the L1 ball ``2t^2 + 2t + 1`` on the von Neumann
    torus and, with ``moore=True``, the Chebyshev square ``(2t + 1)^2``.
    """
    t = np.asarray(t)
    if moore and dim != 2:
        raise ValueError("The Moore cone is a 2-D notion; pass dim=2.")
    if dim == 1:
        return 2 * t + 1
    if dim == 2:
        return (2 * t + 1) ** 2 if moore else 2 * t * t + 2 * t + 1
    raise ValueError(f"Cones are defined for dim 1 (ring) and 2 (von Neumann torus), got {dim}.")


def observed_cone(r, dim: int, moore: bool = False):
    """Cells inside the cone the damage actually reached (radius ``r``)."""
    return maximal_cone(r, dim, moore)


def _distance_map(shape: Tuple[int, ...], flip: Tuple[int, ...], moore: bool = False) -> NDArray[np.int_]:
    """L1 (or, for Moore, Chebyshev) distance of every cell from ``flip``, the short way round."""
    dist = np.zeros(shape, dtype=int)
    for axis, (L, x0) in enumerate(zip(shape, flip)):
        d = np.abs(np.arange(L) - x0)
        d = np.minimum(d, L - d)
        d = np.expand_dims(d, tuple(a for a in range(len(shape)) if a != axis))
        dist = np.maximum(dist, d) if moore else dist + d
    return dist


def damage_series(
    step: Callable[[NDArray], NDArray], s0: NDArray, flip: Index, T: int, moore: bool = False
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Hamming distance ``h[t]`` and damage radius ``r[t]``, ``t = 0 .. T``.

    ``s0`` is evolved by ``step`` alongside a copy with the cell ``flip``
    inverted. ``h[t]`` counts the cells that differ at time ``t`` and ``r[t]``
    is the largest distance of a differing cell from ``flip`` (0 when none
    differs): L1 by default, Chebyshev with ``moore=True``, so that in either
    case the radius grows by at most one per step. Every side must satisfy
    ``L >= 2T + 3`` so that the damage never wraps round the torus.
    """
    s0 = np.asarray(s0, dtype=int)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}.")
    flip = (int(flip),) if np.ndim(flip) == 0 else tuple(int(x) for x in flip)
    if len(flip) != s0.ndim:
        raise ValueError(f"flip must index a {s0.ndim}-D configuration, got {flip}.")
    if any(not 0 <= x < L for x, L in zip(flip, s0.shape)):
        raise ValueError(f"flip {flip} lies outside the configuration of shape {s0.shape}.")
    if min(s0.shape) < 2 * T + 3:
        raise ValueError(
            f"Every side must be at least 2T + 3 = {2 * T + 3} cells so the damage cannot "
            f"wrap round the lattice; got shape {s0.shape} for T = {T}."
        )
    dist = _distance_map(s0.shape, flip, moore)
    a = s0.copy()
    b = s0.copy()
    b[flip] ^= 1
    h = np.empty(T + 1, dtype=int)
    r = np.empty(T + 1, dtype=int)
    for t in range(T + 1):
        if t:
            a, b = step(a), step(b)
        diff = a != b
        h[t] = int(diff.sum())
        r[t] = int(dist[diff].max()) if h[t] else 0
    return h, r


def damage_summary(h: NDArray, r: NDArray, dim: int, window: int, moore: bool = False) -> dict:
    """The three normalised views, averaged over the final ``window`` steps.

    The window is ``t = T - window + 1 .. T`` and must not include ``t = 0``,
    where the front speed is undefined. ``moore`` selects the square cone.
    """
    h = np.asarray(h)
    r = np.asarray(r)
    T = h.size - 1
    if not 1 <= window <= T:
        raise ValueError(f"Need 1 <= window <= T = {T}, got window = {window}.")
    t = np.arange(T - window + 1, T + 1)
    ht, rt = h[t], r[t]
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(ht > 0, ht / observed_cone(rt, dim, moore), 0.0)
    return {
        "v_front": float(np.mean(rt / t)),
        "D_norm": float(np.mean(ht / maximal_cone(t, dim, moore))),
        "fill": float(np.mean(fill)),
        "h_final": int(h[-1]),
        "r_final": int(r[-1]),
    }


def _norm(v: NDArray) -> float:
    """Euclidean norm by numpy's pairwise summation, not BLAS ``ddot``.

    ``np.linalg.norm`` hands the dot product to BLAS, whose summation order
    depends on the number of threads, so the last bit of the result (and hence
    of the exponent) would depend on the machine. numpy's own reduction is
    deterministic, which is what lets the caches be recomputed bit for bit.
    """
    return float(np.sqrt(np.sum(v * v)))


def mle_single_vector(
    bands_at: Callable[[NDArray], NDArray],
    apply: Callable[[NDArray, NDArray], NDArray],
    step: Callable[[NDArray], NDArray],
    s0: NDArray,
    T: int,
    burn: int,
    rng: np.random.Generator,
) -> Tuple[float, Optional[int]]:
    """Growth rate of one random tangent vector over steps ``burn + 1 .. T``.

    ``bands_at(state)`` gives the Jacobian bands at a configuration and
    ``apply(bands, v)`` the product ``J v``; ``step`` advances the
    configuration. The starting vector is standard normal (from ``rng``) and is
    renormalised after every step; the exponent is the mean of ``ln |J_t v_t|``
    over the window. Returns ``(mle, t_dead)``: ``t_dead`` is ``None`` unless
    the vector was annihilated exactly, in which case the exponent is ``-inf``
    and ``t_dead`` is the step (1-based) at which the norm became zero.
    """
    if not 0 <= burn < T:
        raise ValueError(f"Need 0 <= burn < T, got burn={burn}, T={T}.")
    state = np.asarray(s0, dtype=int)
    v = rng.standard_normal(state.shape)
    v /= _norm(v)
    total = 0.0
    for t in range(T):
        v = apply(bands_at(state), v)
        norm = _norm(v)
        if norm == 0.0:
            return -np.inf, t + 1
        v /= norm
        if t >= burn:
            total += np.log(norm)
        state = step(state)
    return total / (T - burn), None


def _damage_and_mle(step, bands_at, apply, s0, flip, dim, T, burn, window, rng, moore=False) -> dict:
    h, r = damage_series(step, s0, flip, T, moore)
    out = damage_summary(h, r, dim, window, moore)
    out["mle"], out["t_dead"] = mle_single_vector(bands_at, apply, step, s0, T, burn, rng)
    out["h"], out["r"] = h, r
    return out


def eca_damage_and_mle(
    rule: int, s0: NDArray, T: int, burn: int, window: int, rng: np.random.Generator
) -> dict:
    """Damage summary and single-vector exponent of an ECA on the ring ``s0``.

    The flipped cell is the central one, ``N // 2``. Returns the keys of
    :func:`damage_summary` plus ``mle``, ``t_dead`` and the raw series ``h``, ``r``.
    """
    s0 = np.asarray(s0, dtype=int)
    return _damage_and_mle(
        lambda s: eca_step(rule, s),
        lambda s: eca_gradient_bands(rule, s),
        lambda b, v: apply_bands(b, v[:, None])[:, 0],
        s0, s0.size // 2, 1, T, burn, window, rng,
    )


def ot_damage_and_mle(
    rule: int, s0: NDArray, T: int, burn: int, window: int, rng: np.random.Generator,
    neighbourhood: Neighbourhood = VON_NEUMANN,
) -> dict:
    """The same for an outer-totalistic rule on the torus ``s0``.

    On the Moore neighbourhood (``neighbourhood=MOORE``) the cone is the
    Chebyshev square and the radius the Chebyshev distance.
    """
    s0 = np.asarray(s0, dtype=int)
    L = s0.shape[0]
    return _damage_and_mle(
        lambda s: ot_step(rule, s, neighbourhood),
        lambda s: ot_gradient_bands(rule, s, neighbourhood),
        apply_bands_2d,
        s0, (L // 2, L // 2), 2, T, burn, window, rng,
        moore=neighbourhood == MOORE,
    )
