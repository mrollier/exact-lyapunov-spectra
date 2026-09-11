# Changelog

## 1.1.0 (2026-09-12)

### Corrections to Vichniac (1990), Table 1: five rules, seven entries

- Rule 172 added to the list of misprinted rows (62, 110, 130, 146, 172). Its
  printed `d/dx[i-1]` is the XNOR `x[i] x[i+1] + ~x[i] ~x[i+1]`; the correct
  entry is the XOR `x[i] ~x[i+1] + ~x[i] x[i+1]`.
- Audit finding: the previous release did not store Vichniac's published table
  at all. It hard-coded only the four corrected entries taken from the
  manuscript (`PAPER_CORRECTIONS`) and checked those against the computation,
  so rule 172 was never compared and the omission could not be detected. The
  recomputed 88-row table already contained the correct XOR for 172.
- New `src/lyapunov/vichniac_table1.py`: the published table transcribed
  exactly as printed (verified against the Physica D scan on 12 September
  2026), misprints deliberately kept; the single source of truth for "what
  Vichniac printed". `PAPER_CORRECTIONS`, `check_corrections` and
  `paper_correction_truth_tables` are removed; the corrected values are now
  computed, never transcribed.
- `src/lyapunov/vichniac.py` compares all 3 x 88 entries by truth table and
  reports the mismatches; it also derives gradients from related rows by
  symmetry (Method 3) and additivity (Method 4) for the cross-checks, and
  generates the manuscript's corrected-entries table (`tab:gradient-table`,
  five ECAs).
- `verify_vichniac.py` keeps `--output` and the column layout of
  `eca_gradient_table.csv`, and additionally writes
  `vichniac_table1_computed.csv`, `vichniac_table1_diff.md` and
  `gradient_corrections_table.tex`. Its summary ends with
  `7 mismatching entries in 5 rules: 62, 110, 130, 146, 172` and the exit code
  is non-zero if the mismatch set differs from the documented seven.
- `verification/test_c4_vichniac_gradients.py` rewritten: exact mismatch set,
  additivity and symmetry cross-checks from the paper's own rows, property (v)
  on printed and computed entries (only 130, `d/dx[i+1]` fails), unit-weight
  and affine sanity checks, and a truth-table check of the manuscript table.
- README, `docs/provenance.md` and docstrings updated from "four corrections"
  to five rules and seven entries.

## 1.0.0

Initial release of the reproducibility package.
