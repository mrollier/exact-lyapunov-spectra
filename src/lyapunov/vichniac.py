"""Recompute the ECA Boolean-gradient table and compare it with Vichniac (1990).

Vichniac, G. Y. (1990), "Boolean derivatives on cellular automata", Physica D
45, 63-74, tabulated the Boolean gradients of the 88 non-equivalent elementary
cellular automata (his Table 1). This module

1. computes all 3 x 88 partial derivatives from the definition (Vichniac's
   eq. (5)): ``df/dx_k = f(..., x_k, ...) XOR f(..., NOT x_k, ...)``, evaluated
   exhaustively over the eight neighbourhoods (via :mod:`lyapunov.rules`);
2. compares every entry with the published transcription in
   :mod:`lyapunov.vichniac_table1` BY TRUTH TABLE, never by string; and
3. renders the computed derivatives in minimal disjunctive normal form (DNF)
   with the Quine-McCluskey minimiser, for display only.

The comparison finds seven misprinted entries in five rows (rules 62, 110, 130,
146 and 172). The corrected values are the computed ones; they are never
transcribed.

Truth patterns
--------------
A partial derivative with respect to one variable is, by Vichniac's property
(v), a function of the other two variables only. Each entry is therefore
summarised by a 4-bit *pattern* over the two remaining variables in the order
(1,1), (1,0), (0,1), (0,0): for d/dx[i-1] the pair is (x[i], x[i+1]); for
d/dx[i] it is (x[i-1], x[i+1]); for d/dx[i+1] it is (x[i-1], x[i]).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, List, Sequence, Tuple

from .rules import (
    gradient_truth_tables,
    nonequivalent_ecas,
    is_affine,
    rule_output,
    truth_table as rule_truth_table,
)
from .quine_mccluskey import (
    DASH,
    Implicant,
    minimise,
    minimise_to_implicants,
    implicants_to_truth_table,
)
from .vichniac_table1 import (  # noqa: F401 -- PROVENANCE is re-exported for the tests
    PUBLISHED,
    MISPRINTS,
    PROVENANCE,
    ADDITIVITY,
    published_row,
)

# Manuscript variable names (juxtaposition = AND, + = OR, ! = NOT).
VAR_NAMES = ("s_{i-1}", "s_i", "s_{i+1}")
SLOT_NAMES = ("d/dx[i-1]", "d/dx[i]", "d/dx[i+1]")
# Compact letters of Vichniac's Table 1 transcription, per variable index.
COMPACT_LETTERS = ("L", "C", "R")
# The two "other" variables of each derivative slot.
OTHERS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
# Column order of a 4-bit pattern over the two other variables.
PAIRS = ((1, 1), (1, 0), (0, 1), (0, 0))


# ---------------------------------------------------------------------------
# Compact-notation parsing and truth patterns
# ---------------------------------------------------------------------------
def eval_compact(expr: str, l: int, c: int, r: int) -> int:
    """Evaluate a compact Table 1 entry such as ``'Cr+cR'`` at ``(l, c, r)``."""
    expr = expr.strip()
    if expr == "0":
        return 0
    if expr == "1":
        return 1
    val = {"L": l, "C": c, "R": r, "l": 1 - l, "c": 1 - c, "r": 1 - r}
    return int(any(all(val[ch] for ch in term) for term in expr.split("+")))


def compact_truth_table(expr: str) -> List[int]:
    """8-entry truth table (index ``4l + 2c + r``) of a compact expression."""
    return [eval_compact(expr, l, c, r) for l, c, r in product((0, 1), repeat=3)]


def compact_mentions_own_variable(expr: str, slot: int) -> bool:
    """True if a printed entry for ``slot`` contains the variable it is
    differentiated with respect to (a violation of Vichniac's property (v))."""
    letter = COMPACT_LETTERS[slot]
    return letter in expr or letter.lower() in expr


def pattern4(tt8: Sequence[int], slot: int) -> str:
    """4-bit pattern of an 8-entry truth table restricted to the two variables
    other than ``slot``, columns ordered as in :data:`PAIRS`.

    Raises ``ValueError`` if the table depends on the slot's own variable, since
    then no 4-bit pattern exists.
    """
    a, b = OTHERS[slot]
    bits = []
    for va, vb in PAIRS:
        x = [0, 0, 0]
        x[a], x[b] = va, vb
        x0 = x.copy()
        x0[slot] = 0
        x1 = x.copy()
        x1[slot] = 1
        v0 = tt8[4 * x0[0] + 2 * x0[1] + x0[2]]
        v1 = tt8[4 * x1[0] + 2 * x1[1] + x1[2]]
        if v0 != v1:
            raise ValueError(f"Table depends on its own variable {SLOT_NAMES[slot]}.")
        bits.append(str(v0))
    return "".join(bits)


def pattern4_lenient(tt8: Sequence[int], slot: int) -> str:
    """As :func:`pattern4` but, for a *printed* entry that illegally depends on
    its own variable, evaluate with that variable set to 0 rather than fail.
    Used only to display the 130 misprint; the comparison still fails it."""
    a, b = OTHERS[slot]
    bits = []
    for va, vb in PAIRS:
        x = [0, 0, 0]
        x[a], x[b] = va, vb
        bits.append(str(tt8[4 * x[0] + 2 * x[1] + x[2]]))
    return "".join(bits)


def rule_table_string(rule: int) -> str:
    """The 8-bit Wolfram string of ``rule`` (leftmost digit is f(1,1,1))."""
    return f"{rule:08b}"


def computed_patterns(rule: int) -> Tuple[str, str, str]:
    """4-bit patterns of the three derivatives computed from definition (5)."""
    gtt = gradient_truth_tables(rule)
    return tuple(pattern4(list(gtt[k]), k) for k in range(3))  # type: ignore[return-value]


def published_patterns(rule: int) -> Tuple[str, str, str]:
    """4-bit patterns of the three printed entries of ``rule``.

    An entry that depends on its own variable (only 130, d/dx[i+1]) is
    displayed with that variable set to 0; :func:`compare_table` treats such an
    entry as a mismatch regardless of the displayed pattern.
    """
    return tuple(  # type: ignore[return-value]
        pattern4_lenient(compact_truth_table(expr), k)
        for k, expr in enumerate(published_row(rule))
    )


def published_entry_is_legal(rule: int, slot: int) -> bool:
    """True if the printed entry does not depend on its own variable."""
    tt = compact_truth_table(published_row(rule)[slot])
    try:
        pattern4(tt, slot)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Rendering (display only): compact notation and manuscript notation
# ---------------------------------------------------------------------------
def _ordered_literals(imp: Implicant) -> List[Tuple[int, int]]:
    """Literals of an implicant as (negated, index), positive literals first."""
    lits = [(0 if v == 1 else 1, j) for j, v in enumerate(imp) if v != DASH]
    return sorted(lits)


def format_implicants(
    implicants: Sequence[Implicant],
    literal: Callable[[int, bool], str],
    and_sym: str = "",
    or_sym: str = "+",
) -> str:
    """Render implicants as a DNF with a fixed literal convention: within a
    product, positive literals precede negated ones (each group in variable
    order); products are ordered by that same key."""
    if not implicants:
        return "0"
    terms = []
    for imp in implicants:
        lits = _ordered_literals(imp)
        if not lits:
            return "1"
        terms.append((tuple(lits), and_sym.join(literal(j, bool(neg)) for neg, j in lits)))
    return or_sym.join(t for _, t in sorted(terms))


def _compact_literal(j: int, negated: bool) -> str:
    return COMPACT_LETTERS[j].lower() if negated else COMPACT_LETTERS[j]


def _text_literal(j: int, negated: bool) -> str:
    names = ("s[i-1]", "s[i]", "s[i+1]")
    return ("~" if negated else "") + names[j]


def _latex_literal(j: int, negated: bool) -> str:
    subs = ("i-1", "i", "i+1")
    return (r"\bar{s}_{%s}" if negated else r"s_{%s}") % subs[j]


def compact_dnf(tt8: Sequence[int]) -> str:
    """Minimal DNF of a truth table in the compact Table 1 notation."""
    return format_implicants(minimise_to_implicants(list(tt8), 3), _compact_literal)


def manuscript_dnf(tt8: Sequence[int], style: str = "text") -> str:
    """Minimal DNF in manuscript notation (``text`` uses ``~`` for NOT and
    ``s[i]``; ``latex`` uses an overbar)."""
    lit = {"text": _text_literal, "latex": _latex_literal}[style]
    return format_implicants(minimise_to_implicants(list(tt8), 3), lit, or_sym=" + ")


def computed_compact(rule: int) -> Tuple[str, str, str]:
    """The three computed derivatives of ``rule`` in compact notation."""
    gtt = gradient_truth_tables(rule)
    return tuple(compact_dnf(list(gtt[k])) for k in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Comparison of the computed table with the published one
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mismatch:
    rule: int
    table: str
    slot: int
    printed: str
    printed_pattern: str
    correct_pattern: str
    correct: str

    @property
    def slot_name(self) -> str:
        return SLOT_NAMES[self.slot]


def compare_table(rules: Iterable[int] | None = None) -> List[Mismatch]:
    """Compare every published entry with the computed derivative by truth table.

    Returns the list of mismatching entries in ascending (rule, slot) order.
    A printed entry that depends on its own variable is always a mismatch.
    """
    if rules is None:
        rules = sorted(PUBLISHED)
    out: List[Mismatch] = []
    for rule in rules:
        correct_p = computed_patterns(rule)
        printed_p = published_patterns(rule)
        correct_e = computed_compact(rule)
        for k, printed in enumerate(published_row(rule)):
            legal = published_entry_is_legal(rule, k)
            if legal and printed_p[k] == correct_p[k]:
                continue
            out.append(Mismatch(rule, rule_table_string(rule), k, printed,
                                printed_p[k], correct_p[k], correct_e[k]))
    return out


def mismatch_summary(mismatches: Sequence[Mismatch]) -> str:
    """Console summary line, e.g. ``7 mismatching entries in 5 rules: 62, ...``."""
    rules = sorted({m.rule for m in mismatches})
    return (f"{len(mismatches)} mismatching entries in {len(rules)} rules: "
            + ", ".join(str(r) for r in rules))


def diff_markdown(mismatches: Sequence[Mismatch]) -> str:
    """Markdown report with one block per mismatching entry."""
    lines = [
        "# Vichniac (1990), Table 1: entries that differ from the computed gradient",
        "",
        "Computed from the definition of the Boolean derivative, eq. (5) of the",
        "paper, exhaustively over the eight neighbourhoods; compared by truth",
        "table. Patterns are over the two remaining variables in the order",
        "(1,1), (1,0), (0,1), (0,0). Notation: L = x[i-1], C = x[i], R = x[i+1];",
        "lower case = complemented literal; juxtaposition = AND; + = OR.",
        "",
    ]
    for m in mismatches:
        illegal = "" if published_entry_is_legal(m.rule, m.slot) else (
            "  (illegal: contains the variable it is differentiated with"
            " respect to; shown with that variable set to 0)")
        lines += [
            f"## Rule {m.rule} ({m.table}), {m.slot_name}",
            "",
            f"- printed expression: `{m.printed}`",
            f"- printed pattern:    `{m.printed_pattern}`{illegal}",
            f"- correct pattern:    `{m.correct_pattern}`",
            f"- correct expression: `{m.correct}`",
            "",
        ]
    lines.append(mismatch_summary(mismatches))
    lines.append("")
    return "\n".join(lines)


def computed_table_rows(rules: Iterable[int] | None = None) -> List[dict]:
    """Rows of ``vichniac_table1_computed.csv``: rule, 8-bit table, and the
    three computed derivatives (compact notation) with their 4-bit patterns."""
    if rules is None:
        rules = sorted(PUBLISHED)
    rows = []
    for rule in rules:
        e = computed_compact(rule)
        p = computed_patterns(rule)
        rows.append({
            "rule": rule,
            "table": rule_table_string(rule),
            "dL": e[0], "dC": e[1], "dR": e[2],
            "dL_pattern": p[0], "dC_pattern": p[1], "dR_pattern": p[2],
        })
    return rows


# ---------------------------------------------------------------------------
# Independent cross-checks that use only Vichniac's own table
# ---------------------------------------------------------------------------
def additivity_patterns(rule: int) -> Tuple[str, str, str]:
    """Method 4: the gradient of ``rule = n1 XOR n2`` as the XOR of the
    PUBLISHED rows of n1 and n2 (from :data:`ADDITIVITY`)."""
    n1, n2 = ADDITIVITY[rule]
    if n1 ^ n2 != rule:
        raise ValueError(f"{n1} XOR {n2} != {rule}")
    p1, p2 = published_patterns(n1), published_patterns(n2)
    return tuple(  # type: ignore[return-value]
        "".join(str(int(a) ^ int(b)) for a, b in zip(x, y)) for x, y in zip(p1, p2)
    )


def _sigma(l: int, c: int, r: int, complemented: Sequence[str], reflect: bool) -> List[int]:
    y = [l, c, r]
    for v in complemented:
        y[COMPACT_LETTERS.index(v)] ^= 1
    if reflect:
        y = [y[2], y[1], y[0]]
    return y


def substituted_row(source_rule: int, complemented: Sequence[str], reflect: bool) -> List[List[int]]:
    """Method 3: derive a target row from the PUBLISHED row of ``source_rule``.

    If ``f_target(x) = f_source(sigma(x)) XOR out``, where ``sigma`` complements
    the listed target variables (letters L, C, R) and, if ``reflect``, swaps L
    and R, then the target's derivative in slot ``k`` equals the source's
    derivative in slot ``pi(k)`` evaluated at ``sigma(x)``, with ``pi`` the
    swap of the outer slots when reflecting. Output complementation does not
    affect derivatives. Returns three 8-entry truth tables over the target's
    ``(L, C, R)``.
    """
    src = published_row(source_rule)
    tables = [[0] * 8 for _ in range(3)]
    for l, c, r in product((0, 1), repeat=3):
        y = _sigma(l, c, r, complemented, reflect)
        for k in range(3):
            pk = 2 - k if reflect else k
            tables[k][4 * l + 2 * c + r] = eval_compact(src[pk], *y)
    return tables


def transformed_rule(source_rule: int, complemented: Sequence[str], reflect: bool, out: int) -> int:
    """The rule number of ``f(x) = f_source(sigma(x)) XOR out`` (see
    :func:`substituted_row`)."""
    n = 0
    for l, c, r in product((0, 1), repeat=3):
        y = _sigma(l, c, r, complemented, reflect)
        n |= (rule_output(source_rule, *y) ^ out) << (4 * l + 2 * c + r)
    return n


# ---------------------------------------------------------------------------
# Manuscript artefacts
# ---------------------------------------------------------------------------
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


def manuscript_correction_rows(rules: Iterable[int] | None = None, style: str = "text") -> List[dict]:
    """Rows of the manuscript's corrected-entries table (``tab:gradient-table``):
    rule, minimal DNF of phi, and the three corrected gradient components, all
    computed. Literal convention: positive literals before barred ones."""
    if rules is None:
        rules = sorted(MISPRINTS)
    rows = []
    for rule in rules:
        gtt = gradient_truth_tables(rule)
        rows.append({
            "rule": rule,
            "phi": manuscript_dnf(list(rule_truth_table(rule)), style),
            "grad_left": manuscript_dnf(list(gtt[0]), style),
            "grad_centre": manuscript_dnf(list(gtt[1]), style),
            "grad_right": manuscript_dnf(list(gtt[2]), style),
        })
    return rows


def manuscript_correction_table_latex() -> str:
    """LaTeX source of the corrected-entries table (five ECAs)."""
    rows = manuscript_correction_rows(style="latex")
    lines = [
        r"% Generated by scripts/verify_vichniac.py; do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Corrected entries of Table~1 of Vichniac (1990) for the five ECAs",
        r"whose printed gradient contains a misprint (seven entries in total).",
        r"Rule and gradient are given as minimal disjunctive normal forms; an",
        r"overbar denotes negation.}",
        r"\label{tab:gradient-table}",
        r"\begin{tabular}{rll}",
        r"\hline",
        r"Rule & $\phi$ & $\nabla\phi$ \\",
        r"\hline",
    ]
    for r in rows:
        grad = ", ".join((r["grad_left"], r["grad_centre"], r["grad_right"]))
        lines.append(f"{r['rule']} & ${r['phi']}$ & $({grad})$ \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_gradient_table(rules: Sequence[int] | None = None) -> List[dict]:
    """Assemble the 88-rule gradient table in manuscript notation.

    Defaults to the 88 non-equivalent ECAs. Each row records the rule, its update
    function and gradient in DNF, whether the rule is affine (constant Jacobian),
    and a self-consistency flag verifying that the minimised DNF of each gradient
    reproduces its truth table.
    """
    if rules is None:
        rules = nonequivalent_ecas()
    table = []
    for rule in rules:
        gtt = gradient_truth_tables(rule)
        dnf = gradient_dnf(rule)
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
