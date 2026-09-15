"""A small Quine-McCluskey Boolean minimiser.

Used to print the ECA Boolean gradients in disjunctive normal form (DNF) for the
corrected Vichniac table. Given a truth table over ``n`` variables it returns a
minimum sum-of-products cover (fewest products; exact for ``n <= 4``, greedy
beyond); the only property we rely on for correctness is that the cover
reproduces the truth table exactly.

An *implicant* is represented as a tuple of length ``n`` with entries in
``{0, 1, DASH}``, where ``DASH`` marks an eliminated (don't-care) variable. The
variable order matches :mod:`lyapunov.rules`: for ``n = 3`` the minterm index of
``(l, c, r)`` is ``4l + 2c + r``, so position 0 is the most significant bit.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

DASH = -1
Implicant = Tuple[int, ...]
# Up to this many variables the cover of the non-essential minterms is found by
# exhaustive search over subsets of the prime implicants (at most 2**(2**n)
# functions, a handful of primes each); beyond it a greedy cover is used.
EXACT_COVER_MAX_VARS = 4


def _minterm_bits(m: int, n: int) -> Implicant:
    return tuple((m >> (n - 1 - i)) & 1 for i in range(n))


def _covers(impl: Implicant, m: int, n: int) -> bool:
    bits = _minterm_bits(m, n)
    return all(a == DASH or a == b for a, b in zip(impl, bits))


def _combine(a: Implicant, b: Implicant) -> Implicant | None:
    """Combine two implicants that differ in exactly one non-dash position."""
    diff = 0
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        else:
            diff += 1
            out.append(DASH)
        if diff > 1:
            return None
    return tuple(out) if diff == 1 else None


def prime_implicants(minterms: Sequence[int], n: int) -> List[Implicant]:
    """Return all prime implicants covering ``minterms`` (n-variable function)."""
    current = {_minterm_bits(m, n) for m in minterms}
    primes: set[Implicant] = set()
    while current:
        used: set[Implicant] = set()
        combined: set[Implicant] = set()
        cur = list(current)
        for a, b in combinations(cur, 2):
            c = _combine(a, b)
            if c is not None:
                combined.add(c)
                used.add(a)
                used.add(b)
        primes |= {imp for imp in current if imp not in used}
        current = combined
    return sorted(primes)


def minimise_to_implicants(truth_table: Sequence[int], n: int) -> List[Implicant]:
    """Return a minimum sum-of-products cover of ``truth_table``.

    Essential prime implicants are taken first. The remaining minterms are then
    covered by the smallest subset of the other primes, found exhaustively when
    ``n <= EXACT_COVER_MAX_VARS`` (the gradient table has ``n = 3``, where a
    greedy choice is one product too long for four of the 256 functions) and
    greedily otherwise. Ties are broken by the sorted order of the primes, so the
    output is deterministic.
    """
    if len(truth_table) != 2 ** n:
        raise ValueError(f"Truth table must have length {2 ** n}, got {len(truth_table)}.")
    minterms = [m for m, v in enumerate(truth_table) if v]
    if not minterms:
        return []  # constant 0
    primes = prime_implicants(minterms, n)

    # Prime-implicant chart: which primes cover each minterm.
    cover = {m: [p for p in primes if _covers(p, m, n)] for m in minterms}

    chosen: List[Implicant] = []
    remaining = set(minterms)

    # Essential prime implicants: the sole cover of some minterm.
    for m, ps in cover.items():
        if len(ps) == 1 and ps[0] not in chosen:
            chosen.append(ps[0])
    remaining -= {m for m in minterms if any(p in chosen for p in cover[m])}

    if remaining and n <= EXACT_COVER_MAX_VARS:
        # Exact: the smallest subset of the other primes that covers what is
        # left, in sorted order so ties are deterministic.
        others = [p for p in primes if p not in chosen]
        for size in range(1, len(others) + 1):
            found = next((sub for sub in combinations(others, size)
                          if all(any(_covers(p, m, n) for p in sub) for m in remaining)),
                         None)
            if found is not None:
                chosen.extend(found)
                remaining = set()
                break
    # Greedy fallback: repeatedly pick the prime covering the most remaining minterms.
    while remaining:
        best = max(primes, key=lambda p: sum(1 for m in remaining if _covers(p, m, n)))
        chosen.append(best)
        remaining -= {m for m in remaining if _covers(best, m, n)}

    # De-duplicate while preserving order.
    seen: set[Implicant] = set()
    result = []
    for p in chosen:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def implicants_to_truth_table(implicants: Sequence[Implicant], n: int) -> List[int]:
    """Reconstruct the truth table covered by a set of implicants (for testing)."""
    return [1 if any(_covers(imp, m, n) for imp in implicants) else 0 for m in range(2 ** n)]


def format_dnf(
    implicants: Sequence[Implicant],
    var_names: Sequence[str],
    not_sym: str = "!",
    and_sym: str = "",
    or_sym: str = " + ",
) -> str:
    """Render a set of implicants as a DNF string.

    ``and_sym=""`` gives juxtaposition (as in the manuscript). An empty cover is
    the constant ``"0"``; a cover containing an all-dash implicant is ``"1"``.
    """
    if not implicants:
        return "0"
    terms = []
    for imp in implicants:
        literals = []
        for name, val in zip(var_names, imp):
            if val == 1:
                literals.append(name)
            elif val == 0:
                literals.append(f"{not_sym}{name}")
        if not literals:
            return "1"  # tautology
        terms.append(and_sym.join(literals))
    return or_sym.join(terms)


def minimise(
    truth_table: Sequence[int],
    var_names: Sequence[str],
    not_sym: str = "!",
    and_sym: str = "",
    or_sym: str = " + ",
) -> str:
    """Minimise a truth table straight to a DNF string."""
    n = len(var_names)
    impls = minimise_to_implicants(truth_table, n)
    return format_dnf(impls, var_names, not_sym, and_sym, or_sym)
