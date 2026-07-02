#!/usr/bin/env python
"""Manuscript Figure 1 -- configuration-space difference patterns of ECAs.

Stem: persistent_defect_eca_diff

For each representative ECA a random initial configuration is evolved alongside a
copy with a single central cell flipped; the cell-by-cell mismatch (the true
Boolean difference pattern) is recorded over time. The mismatch is confined to a
cone expanding from the perturbed cell, and the cone's density grows with the
number of nonzero gradient entries. Rules are grouped into Wolfram class IV (top
row) and class III (bottom row), matching the manuscript.

Parameters (fixed): N = 51 cells, T = 100 steps, one central defect, random
initial configuration with the documented seed.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from lyapunov.jacobian import eca_evolve
import _style

# Representative rules, arranged as in the manuscript (2 rows x 4 columns).
CLASS_IV = [54, 110, 147, 124]
CLASS_III = [30, 101, 90, 150]
N_DEFAULT = 51
T_DEFAULT = 100
SEED_DEFAULT = 20240601


def difference_pattern(rule: int, N: int, T: int, seed: int) -> np.ndarray:
    """Boolean difference pattern from a single central defect on a random config."""
    rng = np.random.default_rng(seed)
    s0 = rng.integers(0, 2, size=N)
    s0_defect = s0.copy()
    s0_defect[N // 2] ^= 1
    a = eca_evolve(rule, s0, T)
    b = eca_evolve(rule, s0_defect, T)
    return a ^ b  # shape (T+1, N), 1 where the two configurations differ


def build_figure(N: int, T: int, seed: int, use_tex: bool = False):
    _style.setup_style(use_tex)
    cmap = _style.defect_cmap()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.2))
    for row, (rules, cls) in enumerate([(CLASS_IV, "Class IV"), (CLASS_III, "Class III")]):
        for col, rule in enumerate(rules):
            ax = axes[row, col]
            patt = difference_pattern(rule, N, T, seed)
            ax.imshow(patt, cmap=cmap, interpolation="nearest", aspect="auto",
                      vmin=0, vmax=1)
            ax.set_title(f"Rule {rule}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cls + "\n" + r"Time $\rightarrow$")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output", default=None, help="Output path (default output/<stem>.pdf)")
    p.add_argument("--no-tex", action="store_true", help="Use mathtext, not LaTeX (default).")
    args = p.parse_args(argv)
    fig = build_figure(args.N, args.T, args.seed, use_tex=False)
    path = _style.save(fig, "persistent_defect_eca_diff", args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
