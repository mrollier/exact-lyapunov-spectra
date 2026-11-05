#!/usr/bin/env python
"""Manuscript Figure 1 -- configuration-space difference patterns of ECAs.

Stem: single_defect_eca_diff

For each representative ECA a random initial configuration is evolved alongside a
copy with a single central cell flipped once, at t = 0; the cell-by-cell mismatch
(the true Boolean difference pattern) is recorded over time. The mismatch is
confined to a cone expanding from the perturbed cell until it wraps around the
ring. For the linear rules 90 and 150 the pattern does not depend on the initial
configuration. Rules are grouped into Wolfram class IV (top row) and class III
(bottom row), matching the manuscript.

Parameters (fixed): N = 101 cells, T = 150 steps, one central defect, and seed 3.
A defect can die out early for some rules and seeds, which would leave a panel
uninformative; seed 3 was chosen because the defect survives to T in all eight
panels. The manuscript's file is written by the --tex option.
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from lyapunov.jacobian import eca_evolve
import _style

# Representative rules, arranged as in the manuscript (2 rows x 4 columns).
CLASS_IV = [54, 110, 147, 124]
CLASS_III = [30, 101, 90, 150]
N_DEFAULT = 101
T_DEFAULT = 150
SEED_DEFAULT = 3


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
    mpl.rcParams.update({"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20,
                         "font.family": "sans-serif", "axes.linewidth": 1.0})
    cmap = ListedColormap(["#fff5f0", "#67000d"])
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 8.0))
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
                ax.set_ylabel(cls)
            if col == len(rules) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(r"$\leftarrow$ Time")
    fig.tight_layout(w_pad=0.3)
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output", default=None, help="Output path (default output/<stem>.pdf)")
    p.add_argument("--tex", action="store_true",
                   help="render text with LaTeX (needs a TeX installation; default: mathtext)")
    args = p.parse_args(argv)
    fig = build_figure(args.N, args.T, args.seed, use_tex=args.tex)
    path = _style.save(fig, "single_defect_eca_diff", args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
