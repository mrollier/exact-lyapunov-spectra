#!/usr/bin/env python
"""Single entry point to reproduce the paper's figures, tables and checks.

Usage
-----
    python reproduce.py all      # regenerate everything and run every check
    python reproduce.py quick    # fast, deterministic subset (used by CI)

``all`` regenerates the CSV tables and the 88-rule gradient table, builds the
six manuscript figures (Figs. 1-6, plus panel A of Fig. 3 on its own and the
supplementary figure S1) into ``output/``, executes the two Fig. 3 notebooks
(four more figures, plus their own checks; skipped if the ``notebook`` extra is
absent) and runs the full verification suite. ``quick`` runs the verification
suite and the (fast) table/gradient exports but skips figure rendering.

Figure and table numbers are those of the revised manuscript (14 Sept 2026).
The heavy caches behind Figs. 5 and 6 are committed and are never recomputed
here; see the commands in the comments below.

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
    ("tables (Tab. 3, Tab. 4)", [sys.executable, "scripts/make_tables.py"]),
    ("gradient table + Vichniac comparison (C4, Tab. 2)", [sys.executable, "scripts/verify_vichniac.py"]),
    ("verification suite (pytest, C1-C7 and C9-C12)", [sys.executable, "-m", "pytest", "-q"]),
]

FIGURES = [
    ("graphs for the supplementary figure S1 (seeded)", [sys.executable, "scripts/make_graphs.py"]),
    ("Fig 1 defect cones", [sys.executable, "figures/fig_defect_cones.py"]),
    ("Fig 2 affine ECA spectra", [sys.executable, "figures/fig_eca_spectra.py"]),
    ("Fig 3, panel A alone (original submission)", [sys.executable, "figures/make_benchmark_figure.py"]),
    ("Fig 3 benchmark + Benettin error per k", [sys.executable, "figures/make_convergence_figure.py"]),
    ("Fig 4 2-D parity", [sys.executable, "figures/fig_2d_parity.py"]),
    # Draws from data/damage/damage_mle_{1d,2d,2d_moore}.npz, all committed.
    # Regenerating them takes ~1 min (1-D), ~10 min (vN) and ~37 min (2000
    # sampled Moore classes) on 10 cores:
    #     python scripts/make_damage_mle.py --dim 1 --recompute
    #     python scripts/make_damage_mle.py --dim 2 --recompute
    #     python scripts/make_damage_mle.py --dim 2 --neighbourhood moore --recompute
    ("Fig 5 damage vs MLE", [sys.executable, "figures/fig_damage_vs_mle.py"]),
    # Draws from data/nonaffine/spectra.npz, which is committed. Regenerating it
    # takes hours (see that script) and is deliberately not part of `all`:
    #     python scripts/make_nonaffine_spectra.py --recompute
    ("Fig 6 non-affine ECA spectra", [sys.executable, "figures/fig_nonaffine_spectra.py"]),
    ("Supplementary Fig S1 defect topologies", [sys.executable, "figures/fig_defect_topologies.py"]),
    # The Fig 3 notebooks: the convergence study and the figure notebook. Executes
    # both in place (rewriting the .ipynb files) and asserts their numbers;
    # skipped (exit 0) if the notebook extra is not installed.
    ("Fig 3 notebooks (study + figure)", [sys.executable, "notebooks/execute.py"]),
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
