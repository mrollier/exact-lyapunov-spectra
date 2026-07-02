#!/usr/bin/env python
"""Recompute and export the ECA Boolean-gradient table (claim C4, table T3).

Recomputes the Boolean gradient of every non-equivalent elementary cellular
automaton from first principles, minimises each gradient to disjunctive normal
form with a Quine-McCluskey simplifier, and writes the result to CSV. It also
verifies the four corrected entries (rules 62, 110, 130, 146) against the values
transcribed from the manuscript's Table 1.

Usage
-----
    python verify_vichniac.py --output data/tables/eca_gradient_table.csv

Exit code is non-zero if any of the four corrections fails to reproduce the
first-principles gradient, or if any minimised DNF is inconsistent with its
truth table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from lyapunov.vichniac import build_gradient_table, check_corrections


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/tables/eca_gradient_table.csv",
        help="CSV path for the full 88-rule gradient table.",
    )
    args = parser.parse_args(argv)

    table = build_gradient_table()  # 88 non-equivalent ECAs
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule", "phi_dnf", "grad_left_dnf", "grad_centre_dnf",
        "grad_right_dnf", "affine", "dnf_consistent",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)

    corrections = check_corrections()
    consistent = all(row["dnf_consistent"] for row in table)

    print(f"Wrote {len(table)} non-equivalent ECA gradients to {out}")
    print(f"Affine (constant-Jacobian) rules among them: "
          f"{sum(r['affine'] for r in table)}")
    print("All minimised DNFs consistent with their truth tables:", consistent)
    print("Corrections vs manuscript Table 1 (rules 62, 110, 130, 146):")
    for rule, ok in sorted(corrections.items()):
        print(f"  rule {rule}: {'confirmed' if ok else 'MISMATCH'}")

    ok = consistent and all(corrections.values())
    if not ok:
        print("FAILED: a correction or a DNF was inconsistent.", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
