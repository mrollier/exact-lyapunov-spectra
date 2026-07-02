#!/usr/bin/env python
"""Manuscript Figure 2 -- singular values and Lyapunov spectra of the affine ECAs.

Stem: singular_values_and_lyapunov_spectra_of_constant_J_ECAs_NO_CLASSES

The 16 affine ECAs are organised by gradient weight |grad phi| into four rows.
Left column: the singular values of the constant Jacobian, sigma_k, against the
normalised frequency k/N. Right column: the relative frequency (log scale) of the
Lyapunov exponents Lambda_k = ln(sigma_k). Rules with |grad phi| = 2 have two
distinct singular-value parametrisations (solid and dashed) with the same
limiting spectrum.

Everything comes from the closed form of src/lyapunov/spectra.py; N = 3001.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from lyapunov.spectra import eca_singular_values
import _style

N_DEFAULT = 3001

# Rows: (title, [(rule, linestyle), ...] for the left panel, rule for histogram).
ROWS = [
    ("Wolfram rules 0, 255,  " + r"$|\nabla\phi| = 0$", [(0, "-")], 0),
    ("Wolfram rules 15, 51, 85, 170, 204, 240,  " + r"$|\nabla\phi| = 1$", [(15, "-")], 15),
    ("Wolfram rules 60, 102, 153, 195, 90, 165,  " + r"$|\nabla\phi| = 2$",
     [(60, "-"), (90, "--")], 60),
    ("Wolfram rules 105, 150,  " + r"$|\nabla\phi| = 3$", [(150, "-")], 150),
]


def build_figure(N: int, use_tex: bool = False):
    _style.setup_style(use_tex)
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 6.4),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    k_over_N = np.arange(N) / N
    for row, (title, curves, hist_rule) in enumerate(ROWS):
        ax_sv, ax_hist = axes[row, 0], axes[row, 1]
        # Left: singular values sigma_k vs k/N.
        for rule, ls in curves:
            sv = eca_singular_values(rule, N)
            ax_sv.plot(k_over_N, sv, ls, color=_style.LINE_BLUE, lw=1.2)
        ax_sv.set_title(title, loc="left")
        ax_sv.set_ylim(-0.15, 3.2)
        ax_sv.set_xlim(0, 1)
        ax_sv.set_yticks([0, 1, 2, 3])
        # Right: histogram of the Lyapunov exponents (log relative frequency).
        # Fixed common bins so a degenerate spectrum (all Lambda_k equal, as for
        # |grad phi| = 1) still shows as a visible bar.
        bins = np.linspace(-4.2, 1.2, 45)
        sv = eca_singular_values(hist_rule, N)
        lam = np.log(sv[sv > 0])
        if lam.size:
            ax_hist.hist(lam, bins=bins, weights=np.ones_like(lam) / lam.size,
                         color=_style.LINE_BLUE, edgecolor="white", linewidth=0.2)
        ax_hist.set_yscale("log")
        ax_hist.set_xlim(-4.2, 1.2)
        ax_hist.set_ylim(1e-3, 1.2)
        if row < 3:
            ax_sv.set_xticklabels([])
            ax_hist.set_xticklabels([])

    axes[3, 0].set_xlabel(r"$k/N$")
    axes[3, 1].set_xlabel(r"$\Lambda_k$")
    fig.supylabel("Singular values of $J$", x=0.02)
    axes[1, 1].set_ylabel("Relative frequency")
    axes[1, 1].yaxis.set_label_position("right")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--output", default=None)
    p.add_argument("--no-tex", action="store_true")
    args = p.parse_args(argv)
    fig = build_figure(args.N, use_tex=False)
    path = _style.save(fig, "singular_values_and_lyapunov_spectra_of_constant_J_ECAs_NO_CLASSES", args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
