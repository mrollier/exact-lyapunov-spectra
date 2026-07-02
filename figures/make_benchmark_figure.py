#!/usr/bin/env python
"""Manuscript Figure 3 -- the affine spectrum as a benchmark.

Stem: benchmark_rule150

The Lyapunov spectrum of an affine rule (default rule 150, N = 101), sorted
descending, computed at a common horizon T = 200 by four methods:
  * the exact closed form (black line),
  * Benettin's QR algorithm (blue open circles),
  * direct multiplication in float16 (red triangles),
  * direct multiplication in float64 (blue dots).
An inset zooms into the leading exponents (k <= zoom-k), where the roles reverse:
direct multiplication is exact there even at low precision, while Benettin has
not yet converged.

Usage:
    python figures/make_benchmark_figure.py --rule 150 --N 101 --T 200 --zoom-k 35
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lyapunov.jacobian import eca_jacobian
from lyapunov.benettin import (
    closed_form_spectrum,
    benettin_spectrum,
    direct_multiplication_spectrum,
)
import _style


def _spectra(rule: int, N: int, T: int):
    J = eca_jacobian(rule, N)
    return {
        "closed": closed_form_spectrum(rule, N),
        "benettin": benettin_spectrum(J, T),
        "f16": direct_multiplication_spectrum(J, T, np.float16),
        "f64": direct_multiplication_spectrum(J, T, np.float64),
    }


def _plot(ax, spec, k, markers: bool):
    ax.plot(k, spec["closed"], "-", color=_style.LINE_BLACK, lw=1.2, label="Exact (closed form)")
    ms = 4 if markers else 3
    ax.plot(k, spec["benettin"], "o", mfc="none", mec=_style.LINE_BLUE, ms=ms,
            mew=0.8, label="Benettin")
    ax.plot(k, spec["f16"], "^", color=_style.ACCENT_RED, ms=ms - 1, lw=0,
            label="Direct mult. (float16)")
    ax.plot(k, spec["f64"], ".", color=_style.LINE_BLUE, ms=ms, lw=0,
            label="Direct mult. (float64)")


def build_figure(rule: int, N: int, T: int, zoom_k: int, use_tex: bool = False):
    _style.setup_style(use_tex)
    spec = _spectra(rule, N, T)
    k = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _plot(ax, spec, k, markers=True)
    ax.set_xlabel(r"Exponent index $k$ (sorted)")
    ax.set_ylabel(r"Lyapunov exponent $\Lambda_k$")
    ax.set_xlim(1, N)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    # Inset: leading exponents (k <= zoom_k).
    axins = inset_axes(ax, width="45%", height="42%", loc="lower left",
                       borderpad=1.6)
    kz = k[:zoom_k]
    specz = {key: val[:zoom_k] for key, val in spec.items()}
    _plot(axins, specz, kz, markers=False)
    axins.set_xlim(1, zoom_k)
    axins.set_title(f"detail for $k \\leq {zoom_k}$", fontsize=8)
    axins.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rule", type=int, default=150)
    p.add_argument("--N", type=int, default=101)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--zoom-k", type=int, default=35, dest="zoom_k")
    p.add_argument("--output", default=None)
    p.add_argument("--no-tex", action="store_true")
    args = p.parse_args(argv)
    fig = build_figure(args.rule, args.N, args.T, args.zoom_k, use_tex=False)
    stem = f"benchmark_rule{args.rule}"
    path = _style.save(fig, stem, args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
