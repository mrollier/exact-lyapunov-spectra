"""Table 1 of Vichniac (1990), exactly as printed.

This module is the single source of truth in this repository for *what
Vichniac printed*. It is a transcription of Table 1 of

    Vichniac, G. Y. (1990). Boolean derivatives on cellular automata.
    Physica D 45, 63-74.

transcribed from the Physica D scan and verified against the scan by aligned
rendering and by pixel-level overbar counting on 12 September 2026.

Limitation: the scan is the publisher's copyrighted PDF and is not distributed
with this repository, so the transcription cannot be re-checked from the
repository alone; a reader who wants to verify it needs a copy of the article
(doi:10.1016/0167-2789(90)90174-N). The misprint result does not rest on the
transcription of the misprinted rows themselves: the corrected entries are
computed from the definition of the Boolean derivative and cross-checked by
additivity and symmetry from the correctly printed rows.

The transcription DELIBERATELY contains the seven misprints found in the
published table (five rules: 62, 110, 130, 146 and 172). Do not "fix" these
entries: their purpose is to record what was printed, so that
:mod:`lyapunov.vichniac` can compare the published table with the gradient
computed from the definition of the Boolean derivative. The corrected values
are computed, never transcribed.

Notation (compact, one letter per literal)
------------------------------------------
``L`` = x[i-1], ``C`` = x[i], ``R`` = x[i+1]; lower case is the complemented
literal (``l`` = NOT x[i-1], and so on). Juxtaposition is AND, ``+`` is OR,
``0`` and ``1`` are constants. Each row lists the three partial derivatives
``(d/dx[i-1], d/dx[i], d/dx[i+1])`` in that order, comma separated, with the
literal order exactly as printed (Vichniac's literal order is not consistent
between rows, e.g. row 1 prints ``cr`` and row 152 prints ``rc``). Compare
entries by truth table, never by string.

Rule numbers follow Wolfram's convention: the output for neighbourhood
``(x[i-1], x[i], x[i+1]) = (L, C, R)`` is bit ``4L + 2C + R`` of the rule
number, so the leftmost digit of the 8-bit string is f(1,1,1). The 88 rows are
the minimal representatives of the reflection/complementation classes, in
ascending order, as in the paper.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Rule -> "d/dx[i-1], d/dx[i], d/dx[i+1]" exactly as printed by Vichniac.
PUBLISHED: Dict[int, str] = {
    0: "0, 0, 0",
    1: "cr, lr, cl",
    2: "cR, lR, lc",
    3: "c, l, 0",
    4: "Cr, lr, lC",
    5: "r, 0, l",
    6: "Cr+cR, l, l",
    7: "c+r, lR, lC",
    8: "CR, lR, lC",
    9: "CR+cr, l, l",
    10: "R, 0, l",
    11: "c+R, lr, lC",
    12: "C, l, 0",
    13: "C+r, lR, lc",
    14: "C+R, lr, lc",
    15: "1, 0, 0",
    18: "c, Lr+lR, c",
    19: "cR, l+r, Lc",
    22: "c+r, l+r, l+c",
    23: "Cr+cR, Lr+lR, Lc+lC",
    24: "CR+cr, Lr+lR, Lc+lC",
    25: "CR, l+r, l+c",
    26: "c+R, Lr, l+c",
    27: "R, r, Lc+lC",
    28: "C+r, l+r, Lc",
    29: "C, Lr+lR, c",
    30: "1, r, c",
    32: "cR, LR, Lc",
    33: "c, LR+lr, c",
    34: "0, R, c",
    35: "cr, l+R, Lc",
    36: "Cr+cR, LR+lr, Lc+lC",
    37: "c+r, LR, l+c",
    38: "Cr, l+R, l+c",
    40: "R, R, Lc+lC",
    41: "c+R, l+R, l+c",
    42: "CR, LR, l+c",
    43: "CR+cr, LR+lr, Lc+lC",
    44: "C+R, l+R, Lc",
    45: "1, R, c",
    46: "C, LR+lr, c",
    50: "cr, L+R, lc",
    51: "0, 1, 0",
    54: "r, 1, l",
    56: "c+R, L+R, lC",
    57: "R, 1, l",
    58: "CR+cr, L, l",
    60: "1, 1, 0",
    62: "c+R, L+r, lc",          # misprint in d/dx[i-1] (correct: C+r)
    72: "C, Lr+lR, C",
    73: "C+r, l+r, l+C",
    74: "C+R, Lr, l+C",
    76: "CR, l+r, LC",
    77: "CR+cr, Lr+lR, LC+lc",
    78: "R, r, LC+lc",
    90: "1, 0, 1",
    94: "c+R, lr, L+c",
    104: "C+R, L+R, L+C",
    105: "1, 1, 1",
    106: "C, L, 1",
    108: "R, 1, L",
    110: "cR, L+r, L+C",         # misprints in d/dx[i-1] and d/dx[i+1] (correct: CR, L+c)
    122: "C+r, LR, l+C",
    126: "CR+cr, LR+lr, LC+lc",
    128: "CR, LR, LC",
    130: "R, R, RC+rc",          # misprint in d/dx[i+1]: subscript typo (correct: LC+lc)
    132: "C, LR+lr, C",
    134: "C+R, l+R, l+C",
    136: "0, R, C",
    138: "cR, LR, l+C",
    140: "Cr, l+R, LC",
    142: "Cr+cR, LR+lr, LC+lc",
    146: "C+R, L+R, L+C",        # misprints in d/dx[i-1] and d/dx[i+1] (correct: c+R, L+c)
    150: "1, 1, 1",
    152: "rc, L+R, L+C",
    154: "c, L, 1",
    156: "r, 1, L",
    160: "R, 0, L",
    162: "CR, lR, L+c",
    164: "C+R, lr, L+C",
    168: "cR, lR, L+C",
    170: "0, 0, 1",
    172: "CR+cr, l, L",          # misprint in d/dx[i-1]: XNOR printed for XOR (correct: Cr+cR)
    178: "CR+cr, Lr+lR, LC+lc",
    184: "c, Lr+lR, C",
    200: "Cr, L+R, lC",
    204: "0, 1, 0",
    232: "Cr+cR, Lr+lR, Lc+lC",
}

# The rules whose printed rows contain at least one misprint, and the slots
# (0 = d/dx[i-1], 1 = d/dx[i], 2 = d/dx[i+1]) that are wrong. Established by
# lyapunov.vichniac.compare_table() and enforced by the C4 tests.
MISPRINTS: Dict[int, Tuple[int, ...]] = {
    62: (0,),
    110: (0, 2),
    130: (2,),
    146: (0, 2),
    172: (0,),
}

# Probable origin of each misprint (Vichniac's Method 3: derive a row from a
# related rule by complementing variables and/or reflecting). The tuple is
# (source rule, variables complemented, reflected, output complemented): the
# target rule satisfies f_n(L, C, R) = f_s(sigma(L, C, R)) XOR out, where sigma
# complements the listed variables and, if reflected, swaps L and R.
PROVENANCE: Dict[int, Tuple[int, Tuple[str, ...], bool, int]] = {
    62: (56, ("C", "R"), False, 1),    # f62(L,C,R) = NOT f56(L, ~C, ~R)
    110: (44, ("C",), True, 1),        # f110(L,C,R) = NOT f44(R, ~C, L)
    146: (104, ("C",), False, 0),      # f146(L,C,R) = f104(L, ~C, R)
    172: (78, ("C",), True, 1),        # f172(L,C,R) = NOT f78(R, ~C, L)
}
# Rule 130 has no Method 3 explanation: its printed d/dx[i+1] contains x[i+1]
# itself, which violates Vichniac's property (v). It is a subscript typo.

# Additivity decompositions (Vichniac's Method 4): n = n1 XOR n2 as rule
# numbers, with n1 and n2 rows of Table 1 that are printed correctly.
ADDITIVITY: Dict[int, Tuple[int, int]] = {
    62: (60, 2),
    110: (106, 4),
    130: (128, 2),
    146: (128, 18),
    172: (168, 4),
}


def published_row(rule: int) -> Tuple[str, str, str]:
    """The three printed entries of ``rule``, stripped, in slot order."""
    parts = tuple(p.strip() for p in PUBLISHED[rule].split(","))
    if len(parts) != 3:
        raise ValueError(f"Row {rule} does not have three entries: {PUBLISHED[rule]!r}")
    return parts  # type: ignore[return-value]
