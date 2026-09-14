"""Outer-totalistic rules on the two-dimensional von Neumann neighbourhood.

An outer-totalistic rule on the von Neumann neighbourhood maps the state ``c``
of a cell and the number ``n`` of its four live orthogonal neighbours to the
next state ``f(c, n)``. The table has ``2 x 5 = 10`` entries, so there are
``2**10 = 1024`` rules; they are encoded as integers with bit ``5 c + n`` equal
to ``f(c, n)``, which is the direct analogue of Wolfram's ECA code. The familiar
"B/S" (birth/survival) notation lists the ``n`` with ``f(0, n) = 1`` and those
with ``f(1, n) = 1``.

Symmetries are simpler than for ECAs: rotations and reflections of the lattice
are already absorbed by totality, so the only non-trivial equivalence is the
black-white conjugation ``f'(c, n) = 1 - f(1 - c, 4 - n)``. Its 32 fixed points
give ``(1024 + 32) / 2 = 528`` classes; :func:`nonequivalent_outer_totalistic`
returns the minimal representative of each.

The Boolean Jacobian follows :mod:`lyapunov.nonaffine` exactly. A cell has
five inputs, so the Jacobian at a configuration is a five-band ``L**2 x L**2``
matrix; :func:`ot_gradient_bands` evaluates the five Boolean derivatives cell
by cell and :func:`apply_bands_2d` applies the product ``J v`` with five rolled
multiplications. Flipping the centre changes ``c``; flipping a neighbour with
state ``s_k`` changes the count to ``n + 1 - 2 s_k``. The two parity rules
(``PARITY_VN_INCLUSIVE``, ``PARITY_VN_EXCLUSIVE``) have a constant Jacobian
equal to the torus adjacency of :func:`lyapunov.jacobian.build_torus_parity_jacobian`,
which is how this module is tied to the closed form of :mod:`lyapunov.spectra`.

Every function takes a ``neighbourhood`` argument, ``VON_NEUMANN`` by default.
``MOORE`` is the eight-neighbour family of "Life-like" rules: ``2 x 9 = 18``
bits, ``2**18 = 262 144`` rules encoded with bit ``9 c + n``, conjugation
``f'(c, n) = 1 - f(1 - c, 8 - n)`` with ``2**9 = 512`` fixed points and hence
``131 328`` classes; the Jacobian has nine bands. Conway's Life is
``ot_from_bs("B3/S23", MOORE)`` and the two Moore parity rules
(``PARITY_MOORE_INCLUSIVE``, ``PARITY_MOORE_EXCLUSIVE``) have exponents
``ln 9`` and ``ln 8``.
"""
from __future__ import annotations

from typing import List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray


class Neighbourhood(NamedTuple):
    """The neighbours of a cell, in the order of Jacobian bands ``1..k``.

    Band 0 is the centre. ``(dx, dy)`` means the neighbour at ``(i + dx, j + dy)``.
    A rule of the family is an integer below ``n_rules`` whose bit ``counts * c + n``
    is ``f(c, n)`` for ``n = 0 .. k``.
    """
    name: str
    offsets: Tuple[Tuple[int, int], ...]

    @property
    def k(self) -> int:
        return len(self.offsets)

    @property
    def counts(self) -> int:
        return self.k + 1

    @property
    def n_rules(self) -> int:
        return 1 << (2 * self.counts)

    def bit(self, c: int, n: int) -> int:
        return self.counts * c + n


VON_NEUMANN = Neighbourhood("vn", ((1, 0), (-1, 0), (0, 1), (0, -1)))
MOORE = Neighbourhood("moore", tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)))
NEIGHBOURHOODS = {nb.name: nb for nb in (VON_NEUMANN, MOORE)}

# The four orthogonal neighbours, in the order of Jacobian bands 1..4 (kept
# for callers that predate the Moore family).
NEIGHBOUR_OFFSETS: Tuple[Tuple[int, int], ...] = VON_NEUMANN.offsets
N_RULES = VON_NEUMANN.n_rules


def _parity_rule(nb: Neighbourhood, inclusive: bool) -> int:
    return sum(1 << nb.bit(c, n) for c in (0, 1) for n in range(nb.counts)
               if (c * inclusive + n) % 2)


# f(c, n) = (c + n) mod 2 and f(c, n) = n mod 2: the self-inclusive and the
# self-exclusive parity rule, MLE ln 5 and ln 4 (Eq. 13 of the paper); ln 9
# and ln 8 on the Moore neighbourhood.
PARITY_VN_INCLUSIVE = _parity_rule(VON_NEUMANN, True)
PARITY_VN_EXCLUSIVE = _parity_rule(VON_NEUMANN, False)
PARITY_MOORE_INCLUSIVE = _parity_rule(MOORE, True)
PARITY_MOORE_EXCLUSIVE = _parity_rule(MOORE, False)


def _check_rule(rule: int, nb: Neighbourhood) -> None:
    if not (0 <= int(rule) < nb.n_rules):
        raise ValueError(f"An outer-totalistic rule on the {nb.name} neighbourhood is an "
                         f"integer in 0..{nb.n_rules - 1}, got {rule}.")


def _check_state(state: NDArray) -> NDArray[np.int_]:
    """Validate an ``L x L`` binary torus configuration."""
    state = np.asarray(state, dtype=int)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError(f"The configuration must be a square L x L array, got shape {state.shape}.")
    if state.shape[0] < 3:
        raise ValueError(f"Need L >= 3 cells per side on a torus, got L={state.shape[0]}.")
    if np.any((state != 0) & (state != 1)):
        raise ValueError("The configuration must be binary (entries 0 or 1).")
    return state


def ot_table(rule: int, neighbourhood: Neighbourhood = VON_NEUMANN) -> NDArray[np.int_]:
    """The rule as a ``(2, k + 1)`` table: ``table[c, n] = f(c, n)``."""
    nb = neighbourhood
    _check_rule(rule, nb)
    return np.array([[(int(rule) >> nb.bit(c, n)) & 1 for n in range(nb.counts)]
                     for c in (0, 1)], dtype=int)


def bs_notation(rule: int, neighbourhood: Neighbourhood = VON_NEUMANN) -> str:
    """The ``B../S..`` label: neighbour counts that give birth / let survive."""
    table = ot_table(rule, neighbourhood)
    birth = "".join(str(n) for n in range(neighbourhood.counts) if table[0, n])
    survival = "".join(str(n) for n in range(neighbourhood.counts) if table[1, n])
    return f"B{birth}/S{survival}"


def ot_from_bs(label: str, neighbourhood: Neighbourhood = VON_NEUMANN) -> int:
    """Inverse of :func:`bs_notation` (``"B13/S024"`` -> the rule integer)."""
    nb = neighbourhood
    try:
        birth, survival = label.split("/")
        assert birth.startswith("B") and survival.startswith("S")
    except (ValueError, AssertionError):
        raise ValueError(f"Expected a label of the form 'B../S..', got {label!r}.") from None
    rule = 0
    for c, digits in ((0, birth[1:]), (1, survival[1:])):
        for d in digits:
            n = int(d)
            if not 0 <= n <= nb.k:
                raise ValueError(f"Neighbour counts run from 0 to {nb.k}, got {d} in {label!r}.")
            rule |= 1 << nb.bit(c, n)
    return rule


def _conjugate_all(nb: Neighbourhood) -> NDArray[np.int64]:
    """``conjugate_rule`` of every rule of the family at once (vectorised)."""
    rules = np.arange(nb.n_rules, dtype=np.int64)
    out = np.zeros_like(rules)
    for c in (0, 1):
        for n in range(nb.counts):
            source = (rules >> nb.bit(1 - c, nb.k - n)) & 1
            out |= (1 - source) << nb.bit(c, n)
    return out


def conjugate_rule(rule: int, neighbourhood: Neighbourhood = VON_NEUMANN) -> int:
    """The black-white conjugate: ``f'(c, n) = 1 - f(1 - c, k - n)``."""
    nb = neighbourhood
    table = ot_table(rule, nb)
    out = 0
    for c in (0, 1):
        for n in range(nb.counts):
            out |= int(1 - table[1 - c, nb.k - n]) << nb.bit(c, n)
    return out


def nonequivalent_outer_totalistic(neighbourhood: Neighbourhood = VON_NEUMANN) -> List[int]:
    """The rules up to conjugation, as the minimal rule of each pair, sorted.

    528 on the von Neumann neighbourhood, 131 328 on the Moore neighbourhood.
    """
    rules = np.arange(neighbourhood.n_rules, dtype=np.int64)
    return [int(r) for r in rules[rules <= _conjugate_all(neighbourhood)]]


def _neighbour(state: NDArray, offset: Tuple[int, int]) -> NDArray:
    """The array whose entry ``[i, j]`` is ``state[i + dx, j + dy]`` (periodic)."""
    dx, dy = offset
    return np.roll(np.roll(state, -dx, axis=0), -dy, axis=1)


def _live_neighbours(state: NDArray, nb: Neighbourhood) -> NDArray[np.int_]:
    return sum(_neighbour(state, off) for off in nb.offsets)


def ot_step(rule: int, state: NDArray, neighbourhood: Neighbourhood = VON_NEUMANN) -> NDArray[np.int_]:
    """One synchronous step on the ``L x L`` torus, vectorised."""
    state = _check_state(state)
    return ot_table(rule, neighbourhood)[state, _live_neighbours(state, neighbourhood)]


def ot_gradient_bands(
    rule: int, state: NDArray, neighbourhood: Neighbourhood = VON_NEUMANN
) -> NDArray[np.int_]:
    """The Boolean-derivative bands of ``rule`` at ``state``, shape ``(k + 1, L, L)``.

    Band 0 is the derivative with respect to the cell itself, bands ``1..k``
    those with respect to the neighbours in the order of the neighbourhood's
    offsets, each evaluated cell by cell at the neighbourhood actually present.
    """
    nb = neighbourhood
    state = _check_state(state)
    table = ot_table(rule, nb)
    n = _live_neighbours(state, nb)
    base = table[state, n]
    bands = np.empty((nb.counts,) + state.shape, dtype=int)
    bands[0] = base ^ table[1 - state, n]
    for k, off in enumerate(nb.offsets, start=1):
        # Flipping a neighbour in state s_k moves the count by 1 - 2 s_k.
        bands[k] = base ^ table[state, n + 1 - 2 * _neighbour(state, off)]
    return bands


def _neighbourhood_of(bands: NDArray) -> Neighbourhood:
    """The neighbourhood a band array belongs to, from its number of bands."""
    for nb in NEIGHBOURHOODS.values():
        if bands.ndim == 3 and bands.shape[0] == nb.counts:
            return nb
    raise ValueError(f"bands must have shape (5, L, L) or (9, L, L), got {bands.shape}.")


def apply_bands_2d(bands: NDArray, v: NDArray) -> NDArray[np.floating]:
    """``J v`` for the banded Jacobian, in ``O(L**2)``.

    ``(J v)[i, j] = b_0[i, j] v[i, j] + sum_k b_k[i, j] v[(i, j) + off_k]``,
    the 2-D counterpart of :func:`lyapunov.nonaffine.apply_bands`. Five bands
    are read as von Neumann, nine as Moore.
    """
    bands = np.asarray(bands, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nb = _neighbourhood_of(bands)
    if v.shape != bands.shape[1:]:
        raise ValueError(f"v must have shape {bands.shape[1:]}, got {v.shape}.")
    out = bands[0] * v
    for k, off in enumerate(nb.offsets, start=1):
        out += bands[k] * _neighbour(v, off)
    return out


def ot_jacobian_at(
    rule: int, state: NDArray, neighbourhood: Neighbourhood = VON_NEUMANN
) -> NDArray[np.int_]:
    """Dense ``L**2 x L**2`` Boolean Jacobian at ``state`` (for checks at small ``L``).

    Cells are flattened row-major, ``(i, j) -> i L + j``, the order of
    :func:`lyapunov.jacobian.build_torus_parity_jacobian`; ``J[a, b]`` is the
    derivative of cell ``a``'s output with respect to cell ``b``'s input.
    """
    bands = ot_gradient_bands(rule, state, neighbourhood)
    L = bands.shape[1]
    J = np.zeros((L * L, L * L), dtype=int)
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    rows = (i * L + j).ravel()
    offsets = ((0, 0),) + neighbourhood.offsets
    for k, (dx, dy) in enumerate(offsets):
        cols = (((i + dx) % L) * L + (j + dy) % L).ravel()
        J[rows, cols] = bands[k].ravel()
    return J
