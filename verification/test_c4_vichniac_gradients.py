"""Claim C4: recompute every ECA Boolean gradient from first principles and
confirm the four Vichniac corrections (rules 62, 110, 130, 146).

The recomputation goes from the rule table to the Boolean derivative directly.
The four corrected entries are checked against the values transcribed from the
manuscript's Table 1. The full 88-rule table is exported by ``verify_vichniac.py``;
here we verify the corrections and that every minimised gradient DNF is logically
consistent with its truth table.

Note on scope: a full row-by-row comparison against Vichniac's original 1990
table would require digitising that table (a manuscript-value task, flagged in
the audit). What is verifiable here without that source is that (a) the four
corrected entries reproduce the true Boolean derivative and (b) the recomputed
table is internally exact.
"""
from lyapunov.rules import gradient_truth_tables, nonequivalent_ecas
from lyapunov.vichniac import (
    check_corrections,
    paper_correction_truth_tables,
    build_gradient_table,
    PAPER_CORRECTIONS,
)


def test_four_corrections_match_first_principles():
    result = check_corrections()
    assert set(result) == {62, 110, 130, 146}
    assert all(result.values()), f"correction mismatch: {result}"


def test_correction_truth_tables_equal_recomputed_gradients():
    for rule in PAPER_CORRECTIONS:
        computed = gradient_truth_tables(rule)
        paper = paper_correction_truth_tables(rule)
        for i in range(3):
            assert list(computed[i]) == paper[i]


def test_corrected_rules_are_not_affine():
    # The four corrected rules have configuration-dependent gradients.
    from lyapunov.rules import is_affine
    for rule in PAPER_CORRECTIONS:
        assert not is_affine(rule)


def test_full_table_has_88_rows_and_is_internally_consistent():
    table = build_gradient_table()
    assert len(table) == 88
    assert all(row["dnf_consistent"] for row in table)
    # exactly the affine rules whose canonical representative is in the 88 set
    n_affine = sum(row["affine"] for row in table)
    assert n_affine >= 1  # at least rule 0's class; sanity that the flag is wired
