#!/usr/bin/env python
"""Manuscript Figure 4 -- singular values and spectra of the 2-D parity rule.

Stem: singular_values_and_log_spectra_2d_parity

The self-inclusive parity rule on a 2-D torus, for three neighbourhoods (von
Neumann, Moore, radius-2 von Neumann). Left column: the singular-value field
sigma_{k,l} over the (k/N, l/N) plane (the modulus of the 2-D DFT of the
neighbourhood stencil). Right column: the relative frequency (log scale) of the
Lyapunov exponents Lambda_{k,l} = ln(sigma_{k,l}). Row titles give the maximal
exponent ln(5), ln(9), ln(13).

Everything comes from the closed form of src/lyapunov/spectra.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from lyapunov.spectra import (
    parity_2d_singular_values,
    parity_2d_mle,
    VON_NEUMANN_2D,
    MOORE_2D,
    VON_NEUMANN_R2_2D,
)
import _style

N_DEFAULT = 201

ROWS = [
    ("von Neumann", VON_NEUMANN_2D, 5),
    ("Moore", MOORE_2D, 9),
    ("radius-2 von Neumann", VON_NEUMANN_R2_2D, 13),
]


def build_figure(N: int, use_tex: bool = False):
    _style.setup_style(use_tex)
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.2),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})
    for row, (name, offsets, mle_n) in enumerate(ROWS):
        ax_map, ax_hist = axes[row, 0], axes[row, 1]
        sigma = parity_2d_singular_values(offsets, N)

        # Left: singular-value heatmap on a logarithmic colour scale.
        floor = 0.05
        im = ax_map.imshow(np.clip(sigma, floor, None), origin="lower",
                           extent=[0, 1, 0, 1], aspect="equal", cmap="RdBu_r",
                           norm=LogNorm(vmin=floor, vmax=mle_n))
        ax_map.set_title(f"{name}:  MLE = ln({mle_n})", loc="left")
        ax_map.set_xlabel(r"$k/N$")
        ax_map.set_ylabel(r"$l/N$")
        cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\sigma_{k,l}$")

        # Right: histogram of the Lyapunov exponents (log relative frequency).
        lam = np.log(sigma[sigma > 0].ravel())
        ax_hist.hist(lam, bins=50, weights=np.ones_like(lam) / lam.size,
                     color=_style.LINE_BLUE, edgecolor="white", linewidth=0.2)
        ax_hist.set_yscale("log")
        ax_hist.set_xlim(-4.2, 3.2)
        ax_hist.set_ylim(1e-3, 1e-1)
        ax_hist.set_xlabel(r"$\Lambda_{k,l}$")
        ax_hist.set_ylabel("Relative frequency")
        ax_hist.yaxis.set_label_position("right")

        assert abs(parity_2d_mle(offsets) - np.log(mle_n)) < 1e-9  # sanity
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--output", default=None)
    p.add_argument("--no-tex", action="store_true")
    args = p.parse_args(argv)
    fig = build_figure(args.N, use_tex=False)
    path = _style.save(fig, "singular_values_and_log_spectra_2d_parity", args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
