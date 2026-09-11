"""Claim C4: recompute every ECA Boolean gradient from first principles and
compare it, entry by entry and by truth table, with Table 1 of Vichniac (1990).

The recomputation goes from the rule table to the Boolean derivative directly
(Vichniac's eq. (5), evaluated exhaustively). The published table is
transcribed in ``lyapunov.vichniac_table1`` with its misprints intact. The
comparison must find exactly seven mismatching entries in five rows: rules 62,
110, 130, 146 and 172.

Two families of cross-checks show that the corrections follow from Vichniac's
own table, without trusting our computation:

* additivity (his property (iii), Method 4): if n = n1 XOR n2 as rule numbers,
  each gradient component of n is the XOR of the corresponding published
  components of n1 and n2;
* symmetry (his property (vi) and reflection, Method 3): each misprinted row
  except 130 is reproduced verbatim from a related, correctly printed row when
  the variable complementation is omitted, which is the probable origin of the
  misprint.

Sanity tests on the computed table cover property (v) (a derivative does not
depend on its own variable), the unit-weight rules, and the affine rules.
"""
import re
from itertools import product

import pytest

from lyapunov.quine_mccluskey import DASH, minimise_to_implicants
from lyapunov.rules import (
    gradient_truth_tables,
    nonequivalent_ecas,
    is_affine,
    reflect_rule,
    complement_rule,
)
from lyapunov.vichniac import (
    ADDITIVITY,
    MISPRINTS,
    PROVENANCE,
    PUBLISHED,
    additivity_patterns,
    build_gradient_table,
    compact_mentions_own_variable,
    compare_table,
    computed_patterns,
    manuscript_correction_rows,
    mismatch_summary,
    pattern4,
    pattern4_lenient,
    published_entry_is_legal,
    published_patterns,
    published_row,
    substituted_row,
    transformed_rule,
)

EXPECTED_SUMMARY = "7 mismatching entries in 5 rules: 62, 110, 130, 146, 172"
EXPECTED_SLOTS = {(r, k) for r, slots in MISPRINTS.items() for k in slots}


# ---------------------------------------------------------------------------
# The table and the comparison
# ---------------------------------------------------------------------------
def test_published_rows_are_the_88_minimal_representatives():
    assert sorted(PUBLISHED) == nonequivalent_ecas()
    assert len(PUBLISHED) == 88
    for rule in PUBLISHED:
        assert len(published_row(rule)) == 3


def test_exactly_seven_mismatches_in_five_rows():
    mismatches = compare_table()
    assert {(m.rule, m.slot) for m in mismatches} == EXPECTED_SLOTS
    assert len(mismatches) == 7
    assert sorted({m.rule for m in mismatches}) == [62, 110, 130, 146, 172]
    assert mismatch_summary(mismatches) == EXPECTED_SUMMARY


def test_exactly_83_rows_match():
    # 88 rows, 5 of which contain a misprint.
    bad_rules = {m.rule for m in compare_table()}
    assert len(set(PUBLISHED) - bad_rules) == 83


def test_centre_entry_is_correct_in_every_row():
    assert all(m.slot != 1 for m in compare_table())


@pytest.mark.parametrize("rule,slot,correct_pattern", [
    (62, 0, "1101"),
    (110, 0, "1000"),
    (110, 2, "1101"),
    (130, 2, "1001"),
    (146, 0, "1011"),
    (146, 2, "1101"),
    (172, 0, "0110"),
])
def test_corrected_patterns(rule, slot, correct_pattern):
    assert computed_patterns(rule)[slot] == correct_pattern
    assert published_patterns(rule)[slot] != correct_pattern or not published_entry_is_legal(rule, slot)


# ---------------------------------------------------------------------------
# Additivity (Method 4), using only published rows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", sorted(ADDITIVITY))
def test_additivity_reproduces_the_correct_gradient(rule):
    n1, n2 = ADDITIVITY[rule]
    assert n1 ^ n2 == rule
    assert n1 in PUBLISHED and n2 in PUBLISHED
    # the two source rows are printed correctly
    assert not any(m.rule in (n1, n2) for m in compare_table())
    assert additivity_patterns(rule) == computed_patterns(rule)


@pytest.mark.parametrize("rule", sorted(ADDITIVITY))
def test_additivity_differs_from_the_printed_row_in_exactly_the_misprinted_slots(rule):
    added = additivity_patterns(rule)
    printed = published_patterns(rule)
    differing = tuple(k for k in range(3)
                      if added[k] != printed[k] or not published_entry_is_legal(rule, k))
    assert differing == MISPRINTS[rule]


# ---------------------------------------------------------------------------
# Provenance (Method 3): symmetries of correctly printed rows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", sorted(PROVENANCE))
def test_symmetry_identity_holds_as_rule_identity(rule):
    source, comp, reflect, out = PROVENANCE[rule]
    assert transformed_rule(source, comp, reflect, out) == rule


@pytest.mark.parametrize("rule", sorted(PROVENANCE))
def test_substituted_published_source_row_gives_the_correct_gradient(rule):
    source, comp, reflect, _ = PROVENANCE[rule]
    derived = substituted_row(source, comp, reflect)
    assert tuple(pattern4(t, k) for k, t in enumerate(derived)) == computed_patterns(rule)


def _verbatim(rule):
    """Source row with the complementation omitted (reflection still applied)."""
    source, _, reflect, _ = PROVENANCE[rule]
    return tuple(pattern4_lenient(t, k) for k, t in enumerate(substituted_row(source, (), reflect)))


@pytest.mark.parametrize("rule", [110, 146, 172])
def test_printed_row_is_the_source_row_without_complementation(rule):
    assert _verbatim(rule) == published_patterns(rule)


def test_rule_62_complementation_omitted_in_first_slot_only():
    source, comp, reflect, _ = PROVENANCE[62]
    verbatim = _verbatim(62)
    correct = tuple(pattern4(t, k) for k, t in enumerate(substituted_row(source, comp, reflect)))
    printed = published_patterns(62)
    assert printed[0] == verbatim[0] and printed[0] != correct[0]
    assert printed[1] == correct[1] and printed[2] == correct[2]


def test_rule_130_is_a_subscript_typo_violating_property_v():
    # The printed d/dx[i+1] of rule 130 contains x[i+1] itself.
    assert 130 not in PROVENANCE
    assert compact_mentions_own_variable(published_row(130)[2], 2)
    assert not published_entry_is_legal(130, 2)


def test_exactly_one_published_entry_contains_its_own_variable():
    offenders = [(rule, k) for rule in PUBLISHED
                 for k, expr in enumerate(published_row(rule))
                 if compact_mentions_own_variable(expr, k)]
    assert offenders == [(130, 2)]


# ---------------------------------------------------------------------------
# Sanity tests on the computed table
# ---------------------------------------------------------------------------
def test_property_v_no_computed_derivative_depends_on_its_own_variable():
    for rule in range(256):
        gtt = gradient_truth_tables(rule)
        for k in range(3):
            pattern4(list(gtt[k]), k)  # raises if it depends on x_k


@pytest.mark.parametrize("rule", [1, 2, 4, 8, 32, 128])
def test_unit_weight_rules_have_single_two_literal_products(rule):
    gtt = gradient_truth_tables(rule)
    for k in range(3):
        implicants = minimise_to_implicants(list(gtt[k]), 3)
        assert len(implicants) == 1
        assert sum(v != DASH for v in implicants[0]) == 2
        assert computed_patterns(rule)[k].count("1") == 1


@pytest.mark.parametrize("rule", [1, 2, 4, 8])
def test_rules_1_2_4_8_match_vichniac_eqs_22_to_26(rule):
    # Vichniac's worked examples, eqs. (22), (23), (24) and (26), are the rows
    # 1, 2, 4 and 8 of his Table 1; those rows are printed correctly.
    assert computed_patterns(rule) == published_patterns(rule)


AFFINE_IN_TABLE = [0, 15, 51, 60, 90, 105, 150, 170, 204]


def test_affine_rows_of_the_table_have_constant_gradients():
    assert [r for r in PUBLISHED if is_affine(r)] == AFFINE_IN_TABLE
    for rule in AFFINE_IN_TABLE:
        assert all(p in ("0000", "1111") for p in computed_patterns(rule))
        assert all(e in ("0", "1") for e in published_row(rule))


def test_symmetric_images_of_affine_rules_have_constant_gradients():
    images = set()
    for rule in AFFINE_IN_TABLE:
        images |= {reflect_rule(rule), complement_rule(rule), 255 - rule,
                   reflect_rule(255 - rule)}
    assert {240, 195, 165, 153, 102, 85} <= images
    for rule in images:
        assert is_affine(rule)
        assert all(p in ("0000", "1111") for p in computed_patterns(rule))
    assert len(images | set(AFFINE_IN_TABLE)) == 16


# ---------------------------------------------------------------------------
# Manuscript artefacts
# ---------------------------------------------------------------------------
def _eval_text(expr, l, c, r):
    """Evaluate a manuscript-notation DNF such as 's[i]~s[i+1] + s[i+1]~s[i]'."""
    expr = expr.strip()
    if expr in ("0", "1"):
        return int(expr)
    val = {"s[i-1]": l, "s[i]": c, "s[i+1]": r}
    literal = re.compile(r"~?s\[i(?:[+-]1)?\]")
    total = 0
    for term in re.split(r"\+(?![^\[]*\])", expr):  # '+' outside brackets
        prod = 1
        for lit in literal.findall(term):
            prod &= (1 - val[lit[1:]]) if lit.startswith("~") else val[lit]
        total |= prod
    return total


def _tt(expr):
    return [_eval_text(expr, l, c, r) for l, c, r in product((0, 1), repeat=3)]


MANUSCRIPT_TABLE = {
    62: ("s[i-1]~s[i] + s[i]~s[i-1] + s[i+1]~s[i-1]",
         ("s[i]+~s[i+1]", "s[i-1]+~s[i+1]", "~s[i-1]~s[i]")),
    110: ("s[i]~s[i+1] + s[i+1]~s[i-1] + s[i+1]~s[i]",
          ("s[i]s[i+1]", "s[i-1]+~s[i+1]", "s[i-1]+~s[i]")),
    130: ("s[i-1]s[i]s[i+1] + s[i+1]~s[i-1]~s[i]",
          ("s[i+1]", "s[i+1]", "s[i-1]s[i]+~s[i-1]~s[i]")),
    146: ("s[i-1]s[i]s[i+1] + s[i-1]~s[i]~s[i+1] + s[i+1]~s[i-1]~s[i]",
          ("s[i+1]+~s[i]", "s[i-1]+s[i+1]", "s[i-1]+~s[i]")),
    172: ("s[i-1]s[i+1] + s[i]~s[i-1]",
          ("s[i]~s[i+1]+s[i+1]~s[i]", "~s[i-1]", "s[i-1]")),
}


def test_manuscript_correction_table_matches_by_truth_table():
    rows = {r["rule"]: r for r in manuscript_correction_rows()}
    assert sorted(rows) == [62, 110, 130, 146, 172]
    for rule, (phi, grad) in MANUSCRIPT_TABLE.items():
        row = rows[rule]
        assert _tt(row["phi"]) == _tt(phi)
        for key, expected in zip(("grad_left", "grad_centre", "grad_right"), grad):
            assert _tt(row[key]) == _tt(expected)


def test_full_table_has_88_rows_and_is_internally_consistent():
    table = build_gradient_table()
    assert len(table) == 88
    assert all(row["dnf_consistent"] for row in table)
    assert sum(row["affine"] for row in table) == len(AFFINE_IN_TABLE)
