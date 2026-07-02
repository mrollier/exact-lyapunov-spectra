"""Elementary cellular automaton (ECA) rule algebra.

An ECA maps a neighbourhood triple ``(l, c, r) = (s_{i-1}, s_i, s_{i+1})`` to a
single output bit. We follow Wolfram's numbering: the output for neighbourhood
``(l, c, r)`` is bit ``4l + 2c + r`` of the rule number, so there are 256 rules.

The central object for the Lyapunov analysis is the **Boolean gradient**: the
triple of Boolean (Vichniac) derivatives ``(dphi/ds_{i-1}, dphi/ds_i,
dphi/ds_{i+1})``. Each derivative answers "does flipping this input flip the
output?" and is in general itself a function of the neighbourhood, so we store it
as an 8-entry truth table over ``(l, c, r)``.

A rule is **affine** exactly when every one of its Boolean derivatives is a
constant (independent of the neighbourhood). Equivalently, its update is a XOR of
a subset of its inputs and a constant. For such rules the Jacobian does not
depend on the configuration -- the property that makes the exact Lyapunov
spectrum possible.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

# All eight neighbourhoods in ascending code order 0..7, as (l, c, r) triples.
_NEIGHBOURHOODS: Tuple[Tuple[int, int, int], ...] = tuple(
    (n >> 2 & 1, n >> 1 & 1, n & 1) for n in range(8)
)


def _check_eca(rule: int) -> None:
    if not isinstance(rule, (int, np.integer)) or not (0 <= int(rule) <= 255):
        raise ValueError(f"An ECA rule must be an integer in 0..255, got {rule!r}.")


def rule_output(rule: int, left: int, centre: int, right: int) -> int:
    """Output bit of ``rule`` for neighbourhood ``(left, centre, right)``."""
    _check_eca(rule)
    code = 4 * (left & 1) + 2 * (centre & 1) + (right & 1)
    return (int(rule) >> code) & 1


def truth_table(rule: int) -> NDArray[np.int_]:
    """The rule's output as an 8-entry table indexed by ``4l + 2c + r``."""
    _check_eca(rule)
    return np.array([(int(rule) >> code) & 1 for code in range(8)], dtype=int)


def gradient_truth_tables(rule: int) -> NDArray[np.int_]:
    """Boolean derivatives of ``rule`` w.r.t. each input.

    Returns an array of shape ``(3, 8)``. Row 0 is ``dphi/ds_{i-1}`` (left),
    row 1 is ``dphi/ds_i`` (centre), row 2 is ``dphi/ds_{i+1}`` (right). Column
    ``4l + 2c + r`` holds the derivative evaluated at neighbourhood ``(l, c, r)``:
    1 if flipping that input flips the output there, else 0.
    """
    _check_eca(rule)
    gtt = np.zeros((3, 8), dtype=int)
    for l, c, r in _NEIGHBOURHOODS:
        code = 4 * l + 2 * c + r
        base = rule_output(rule, l, c, r)
        gtt[0, code] = base ^ rule_output(rule, l ^ 1, c, r)
        gtt[1, code] = base ^ rule_output(rule, l, c ^ 1, r)
        gtt[2, code] = base ^ rule_output(rule, l, c, r ^ 1)
    return gtt


def is_affine(rule: int) -> bool:
    """True iff every Boolean derivative of ``rule`` is constant (affine rule)."""
    gtt = gradient_truth_tables(rule)
    return bool(np.all(gtt[:, :1] == gtt))  # every column equals the first


# ``has_constant_jacobian`` is the same property under a name that matches the
# paper's phrasing: an affine rule is exactly one with a constant Boolean Jacobian.
has_constant_jacobian = is_affine


def affine_gradient(rule: int) -> Tuple[int, int, int]:
    """The constant gradient ``(a_-, a_o, a_+)`` of an affine rule.

    Raises ``ValueError`` if the rule is not affine (its derivatives are not
    constant, so no single gradient triple exists).
    """
    if not is_affine(rule):
        raise ValueError(f"Rule {rule} is not affine; its gradient is not constant.")
    gtt = gradient_truth_tables(rule)
    return (int(gtt[0, 0]), int(gtt[1, 0]), int(gtt[2, 0]))


def gradient_weight(rule: int) -> int:
    """Number of inputs an affine rule is sensitive to (the gradient weight)."""
    return sum(affine_gradient(rule))


def affine_ecas() -> List[int]:
    """Sorted list of the 16 affine ECAs."""
    return [r for r in range(256) if is_affine(r)]


def reflect_rule(rule: int) -> int:
    """The left-right mirror of ``rule`` (swap the left and right neighbours)."""
    _check_eca(rule)
    out = 0
    for l, c, r in _NEIGHBOURHOODS:
        bit = rule_output(rule, r, c, l)  # mirrored neighbourhood
        out |= bit << (4 * l + 2 * c + r)
    return out


def complement_rule(rule: int) -> int:
    """The black-white (0<->1) conjugate of ``rule``."""
    _check_eca(rule)
    out = 0
    for l, c, r in _NEIGHBOURHOODS:
        bit = 1 - rule_output(rule, 1 - l, 1 - c, 1 - r)
        out |= bit << (4 * l + 2 * c + r)
    return out


def equivalence_orbit(rule: int) -> set:
    """The orbit of ``rule`` under reflection and complementation."""
    _check_eca(rule)
    orbit = set()
    frontier = {rule}
    while frontier:
        orbit |= frontier
        new = set()
        for x in frontier:
            for y in (reflect_rule(x), complement_rule(x)):
                if y not in orbit:
                    new.add(y)
        frontier = new
    return orbit


def nonequivalent_ecas() -> List[int]:
    """The 88 non-equivalent ECAs, as the minimal rule of each symmetry orbit."""
    seen = set()
    reps = []
    for rule in range(256):
        if rule in seen:
            continue
        orbit = equivalence_orbit(rule)
        seen |= orbit
        reps.append(min(orbit))
    return sorted(reps)
