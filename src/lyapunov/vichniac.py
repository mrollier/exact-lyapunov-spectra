"""Recompute the ECA Boolean-gradient table from first principles.

Vichniac tabulated the Boolean gradients of the elementary cellular automata.
The paper recomputes them directly from the rule tables and reports four entries
(rules 62, 110, 130, 146) that differ from the published values and appear to be
misprints. This module performs that recomputation: for any ECA it derives the
gradient truth tables (via :mod:`lyapunov.rules`) and renders them in disjunctive
normal form (via :mod:`lyapunov.quine_mccluskey`).

The four corrected entries are encoded here directly from the manuscript's
Table 1, as Boolean functions of ``(l, c, r) = (s_{i-1}, s_i, s_{i+1})``, so the
first-principles computation can be checked against them.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from .rules import (
    gradient_truth_tables,
    nonequivalent_ecas,
    is_affine,
    truth_table as rule_truth_table,
)
from .quine_mccluskey import minimise, minimise_to_implicants, implicants_to_truth_table

# Manuscript variable names (juxtaposition = AND, + = OR, ! = NOT).
VAR_NAMES = ("s_{i-1}", "s_i", "s_{i+1}")


def _truth_table_from_fn(fn: Callable[[int, int, int], int]) -> List[int]:
    """Build an 8-entry truth table (index 4l+2c+r) from a Boolean function."""
    tt = [0] * 8
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                tt[4 * l + 2 * c + r] = int(fn(l, c, r))
    return tt


# ---------------------------------------------------------------------------
# The four corrected gradient entries, transcribed from Table 1 of the paper.
# Each rule maps to (d/ds_{i-1}, d/ds_i, d/ds_{i+1}) as functions of (l, c, r).
# ---------------------------------------------------------------------------
PAPER_CORRECTIONS: Dict[int, Tuple[Callable, Callable, Callable]] = {
    62: (
        lambda l, c, r: c | (1 - r),          # s_i + !s_{i+1}
        lambda l, c, r: l | (1 - r),          # s_{i-1} + !s_{i+1}
        lambda l, c, r: (1 - l) & (1 - c),    # !s_{i-1} !s_i
    ),
    110: (
        lambda l, c, r: c & r,                # s_i s_{i+1}
        lambda l, c, r: l | (1 - r),          # s_{i-1} + !s_{i+1}
        lambda l, c, r: l | (1 - c),          # s_{i-1} + !s_i
    ),
    130: (
        lambda l, c, r: r,                    # s_{i+1}
        lambda l, c, r: r,                    # s_{i+1}
        lambda l, c, r: (l & c) | ((1 - l) & (1 - c)),  # s_{i-1}s_i + !s_{i-1}!s_i
    ),
    146: (
        lambda l, c, r: r | (1 - c),          # s_{i+1} + !s_i
        lambda l, c, r: l | r,                # s_{i-1} + s_{i+1}
        lambda l, c, r: l | (1 - c),          # s_{i-1} + !s_i
    ),
}


def paper_correction_truth_tables(rule: int) -> List[List[int]]:
    """The three gradient truth tables implied by the manuscript's corrected DNF."""
    fns = PAPER_CORRECTIONS[rule]
    return [_truth_table_from_fn(fn) for fn in fns]


def check_corrections() -> Dict[int, bool]:
    """Confirm each corrected entry equals the first-principles Boolean gradient.

    Returns a mapping ``rule -> True/False``; ``True`` means the manuscript's
    corrected DNF reproduces the gradient computed straight from the rule table.
    """
    result = {}
    for rule, _ in PAPER_CORRECTIONS.items():
        computed = gradient_truth_tables(rule)              # shape (3, 8)
        paper = paper_correction_truth_tables(rule)         # list of 3 x 8
        result[rule] = all(
            list(computed[i]) == paper[i] for i in range(3)
        )
    return result


def gradient_dnf(rule: int) -> Tuple[str, str, str]:
    """Minimised DNF of the three Boolean derivatives of ``rule``."""
    gtt = gradient_truth_tables(rule)
    return tuple(
        minimise(list(gtt[i]), var_names=VAR_NAMES, not_sym="!", and_sym="", or_sym=" + ")
        for i in range(3)
    )


def rule_dnf(rule: int) -> str:
    """Minimised DNF of the rule's own update function ``phi``."""
    return minimise(
        list(rule_truth_table(rule)),
        var_names=VAR_NAMES, not_sym="!", and_sym="", or_sym=" + ",
    )


def build_gradient_table(rules: Sequence[int] | None = None) -> List[dict]:
    """Assemble the gradient table.

    Defaults to the 88 non-equivalent ECAs. Each row records the rule, its update
    function and gradient in DNF, the gradient weight range, and whether the rule
    is affine (constant Jacobian). The self-consistency flag verifies that the
    minimised DNF of each gradient reproduces its truth table.
    """
    if rules is None:
        rules = nonequivalent_ecas()
    table = []
    for rule in rules:
        gtt = gradient_truth_tables(rule)
        dnf = gradient_dnf(rule)
        # self-consistency: the minimised cover must reproduce the truth table
        consistent = all(
            implicants_to_truth_table(minimise_to_implicants(list(gtt[i]), 3), 3)
            == list(gtt[i])
            for i in range(3)
        )
        table.append(
            {
                "rule": rule,
                "phi_dnf": rule_dnf(rule),
                "grad_left_dnf": dnf[0],
                "grad_centre_dnf": dnf[1],
                "grad_right_dnf": dnf[2],
                "affine": is_affine(rule),
                "dnf_consistent": consistent,
            }
        )
    return table
