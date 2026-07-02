#!/usr/bin/env python
"""Single entry point to reproduce the paper's figures, tables and checks.

Usage
-----
    python reproduce.py all      # regenerate everything and run every check
    python reproduce.py quick    # fast, deterministic subset (used by CI)

``all`` regenerates the seeded graphs and CSV tables, exports the 88-rule
gradient table, builds the five manuscript figures into ``output/``, and runs the
full verification suite. ``quick`` runs the verification suite and the (fast)
table/gradient exports but skips figure rendering.

Every step is run as a subprocess so a failure in one is reported without
aborting the summary. The exit code is non-zero if any step fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each step: (label, argv). Run with the current interpreter from ROOT.
TABLES_AND_CHECKS = [
    ("tables (T1, T2)", [sys.executable, "data/make_tables.py"]),
    ("gradient table + corrections (C4/T3)", [sys.executable, "verify_vichniac.py"]),
    ("verification suite (pytest, C1-C7)", [sys.executable, "-m", "pytest", "-q"]),
]

FIGURES = [
    ("graphs (seeded)", [sys.executable, "data/make_graphs.py"]),
    ("Fig 1 defect cones", [sys.executable, "figures/fig_defect_cones.py"]),
    ("Fig 2 affine ECA spectra", [sys.executable, "figures/fig_eca_spectra.py"]),
    ("Fig 3 benchmark", [sys.executable, "figures/make_benchmark_figure.py"]),
    ("Fig 4 2-D parity", [sys.executable, "figures/fig_2d_parity.py"]),
    ("Fig 5 defect topologies", [sys.executable, "figures/fig_defect_topologies.py"]),
]


def run(label: str, argv: list[str]) -> tuple[str, bool, float]:
    start = time.time()
    proc = subprocess.run(argv, cwd=ROOT)
    return label, proc.returncode == 0, time.time() - start


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["all", "quick"], help="what to reproduce")
    args = p.parse_args(argv)

    steps = list(TABLES_AND_CHECKS)
    if args.mode == "all":
        steps = FIGURES + TABLES_AND_CHECKS  # graphs + figures first, then checks

    results = []
    for label, cmd in steps:
        print(f"\n=== {label} ===", flush=True)
        results.append(run(label, cmd))

    print("\n" + "=" * 60)
    print(f"Reproduction summary ({args.mode}):")
    ok_all = True
    for label, ok, dt in results:
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:45s} {dt:6.1f}s")
        ok_all &= ok
    print("=" * 60)
    if not ok_all:
        print("Some steps FAILED.", file=sys.stderr)
        return 1
    print("All steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
