#!/usr/bin/env python
"""Recompute the ECA Boolean-gradient table and compare it with Vichniac (1990).

Computes the Boolean gradient of every one of the 88 non-equivalent elementary
cellular automata from the definition of the Boolean derivative (Vichniac's
eq. (5), evaluated exhaustively over the eight neighbourhoods), compares each
of the 3 x 88 entries BY TRUTH TABLE with Table 1 of Vichniac (1990) as
transcribed in ``lyapunov.vichniac_table1``, and writes:

* ``--output``      the 88-rule gradient table in manuscript notation
                    (unchanged column layout: rule, phi_dnf, grad_*_dnf,
                    affine, dnf_consistent);
* ``--computed``    ``vichniac_table1_computed.csv``: rule, 8-bit table and
                    the three correct entries in Vichniac's compact notation;
* ``--diff``        ``vichniac_table1_diff.md``: one block per mismatching
                    entry (printed expression and pattern, correct pattern and
                    expression);
* ``--corrections`` the LaTeX source of the manuscript's corrected-entries
                    table (``tab:gradient-table``).

The console summary ends with the line
``7 mismatching entries in 5 rules: 62, 110, 130, 146, 172``.

Usage
-----
    python verify_vichniac.py            # from the repository root, as App. A says
    python scripts/verify_vichniac.py [--output data/tables/eca_gradient_table.csv]

The first form runs the launcher at the repository root, which hands over to
this script unchanged; both accept the same arguments. Output paths default
to ``data/tables/`` of the repository, whatever the working directory.

Exit code is non-zero if the set of mismatching entries differs from the seven
documented misprints, or if any minimised DNF is inconsistent with its truth
table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from lyapunov.vichniac import (
    MISPRINTS,
    SLOT_NAMES,
    build_gradient_table,
    compare_table,
    computed_table_rows,
    diff_markdown,
    manuscript_correction_table_latex,
    mismatch_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = REPO_ROOT / "data" / "tables"


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        default=TABLE_DIR / "eca_gradient_table.csv",
        type=Path,
        help="CSV path for the full 88-rule gradient table (manuscript notation).",
    )
    parser.add_argument(
        "--computed",
        default=TABLE_DIR / "vichniac_table1_computed.csv",
        type=Path,
        help="CSV path for the correct Table 1 entries in compact notation.",
    )
    parser.add_argument(
        "--diff",
        default=TABLE_DIR / "vichniac_table1_diff.md",
        type=Path,
        help="Markdown path for the report of mismatching entries.",
    )
    parser.add_argument(
        "--corrections",
        default=TABLE_DIR / "gradient_corrections_table.tex",
        type=Path,
        help="LaTeX path for the manuscript's corrected-entries table.",
    )
    args = parser.parse_args(argv)

    # 1. Full 88-rule table in manuscript notation (backward-compatible layout).
    table = build_gradient_table()
    out = Path(args.output)
    _write_csv(table, out, [
        "rule", "phi_dnf", "grad_left_dnf", "grad_centre_dnf",
        "grad_right_dnf", "affine", "dnf_consistent",
    ])
    consistent = all(row["dnf_consistent"] for row in table)

    # 2. Correct entries in Vichniac's notation, and the truth-table comparison.
    computed = computed_table_rows()
    _write_csv(computed, Path(args.computed), list(computed[0].keys()))
    mismatches = compare_table()
    diff_path = Path(args.diff)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(diff_markdown(mismatches), encoding="utf-8")

    # 3. Manuscript artefact.
    tex_path = Path(args.corrections)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(manuscript_correction_table_latex(), encoding="utf-8")

    # Console summary.
    print(f"Wrote {len(table)} non-equivalent ECA gradients to {out}")
    print(f"Affine (constant-Jacobian) rules among them: "
          f"{sum(r['affine'] for r in table)}")
    print("All minimised DNFs consistent with their truth tables:", consistent)
    print(f"Wrote correct Table 1 entries to {args.computed}")
    print(f"Wrote mismatch report to {args.diff}")
    print(f"Wrote corrected-entries LaTeX table to {args.corrections}")
    print()
    print("Vichniac (1990) Table 1 vs computed gradient (truth-table comparison):")
    for m in mismatches:
        print(f"  rule {m.rule:3d} ({m.table}) {m.slot_name:9s}: printed {m.printed:8s}"
              f" [{m.printed_pattern}]  correct {m.correct:8s} [{m.correct_pattern}]")

    expected = {(r, k) for r, slots in MISPRINTS.items() for k in slots}
    found = {(m.rule, m.slot) for m in mismatches}
    ok = consistent and found == expected
    if not ok:
        print("FAILED: mismatch set differs from the documented misprints, "
              "or a DNF was inconsistent.", file=sys.stderr)
        print(f"  expected: {sorted(expected)}", file=sys.stderr)
        print(f"  found:    {sorted(found)}", file=sys.stderr)
    print(mismatch_summary(mismatches))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
