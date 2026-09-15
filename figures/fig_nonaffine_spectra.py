#!/usr/bin/env python
"""Manuscript Figure 6 -- Lyapunov spectra of nine non-affine ECAs and three affine ones.

Stem: lyapunov_spectra_nonaffine_ecas

The nine rules whose spectra Vispoel et al. (2024) report, at their own settings:
N = 1000 cells on a ring, T = 500 steps, 40 random initial configurations per
rule. Their exponents come from Benettin's QR algorithm averaged over the last
300 steps, after a burn-in of 200, pooled over the 40 samples.

Rules 60, 90 and 150 follow, for comparison. They are affine, so their Jacobian
is constant and their spectrum is the closed form of ``lyapunov.spectra``: one
exact vector, no trajectory and no sampling. Their maximal exponents are exactly
ln 2, ln 2 and ln 3.

Twelve panels in two columns and six rows. Each shows the relative frequency of
the exponents against Lambda_k, on equal-width bins shared by all twelve so the
panels can be read against each other; the vertical scale is logarithmic, as in
Figure 2, and the dashed line marks the maximal exponent. Each panel annotates
that exponent: exactly ln 2 or ln 3 for the three affine rules, and for the nine
others the mean over the 40 samples, carrying the 16th and 84th percentiles of
those samples as an asymmetric spread.

Exponents that are exactly -inf are not drawn, which is the censoring Vispoel
applies when he discards the zero eigenvalues of Y Y^T. Their share is annotated
in each panel. How many there are is a rank question, answered exactly in
``lyapunov.nonaffine.tangent_rank`` (and ``integer_matrix_rank`` for the affine
rules), never by thresholding the QR diagonal.

The affine spectra have much longer left tails than the non-affine ones, because
a singular value passing close to zero sends ln(sigma) down steeply: rule 150
reaches -5.62 against -2.80 for the worst non-affine rule. The shared range is
cut at -2.7 so the bulk of every panel is legible; ``main`` prints how much of
each spectrum falls outside it, and the caption records it.

The heavy computation lives in ``data/make_nonaffine_spectra.py``, which caches
its result; this script only draws. Every layout number is a key of ``STYLE``.

Usage:
    python figures/fig_nonaffine_spectra.py
    python figures/fig_nonaffine_spectra.py --bins 60 --output output/spectra_60bins.pdf
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import _style

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
import make_nonaffine_spectra as run          # noqa: E402  (needs the path above)

STYLE = {
    "figsize": (7.0, 8.2),          # two columns, six shorter rows
    "font_family": "serif",
    "font_serif": ("CMU Serif", "Times New Roman", "DejaVu Serif"),
    # Sized for a figure that spans both columns of the journal's page: at the
    # 7 inch text width the figure is printed about 1:1, so these are the point
    # sizes the reader sees.
    "font_size": 11,
    "label_size": 14,
    "title_size": 12,
    "tick_size": 10,
    "annotation_size": 9.5,
    "bins": 50,
    # The affine tails run to -5.6; cutting here keeps every panel legible and
    # puts the -2 tick near the left edge. main() reports what falls outside.
    "bin_range": (-2.7, 1.2),
    "xticks": (-2, -1, 0, 1),
    # A single exponent in the most populous panel is 1/40000 = 2.5e-5, so an
    # axis cut at 1e-5 still shows the smallest possible bar, and puts the 1e-5
    # tick on the frame.
    # The top decade is empty in every panel and holds the annotation, which
    # the three affine densities would otherwise run into.
    "ylim": (1e-5, 10.0),
    "yticks": (1e-5, 1e-3, 1e-1),
    "bar_colour": _style.LINE_BLUE,
    "bar_edge": "white",
    "bar_edge_width": 0.2,
    "mle_colour": "black",
    "mle_width": 1.0,
    "mle_style": (0, (4, 2)),
    "hspace": 0.45,
    "wspace": 0.12,
    "ylabel_x": 0.035,
}


def _rc(style):
    """rcParams overriding the repo defaults for this figure only."""
    return {
        "font.family": style["font_family"],
        "font.serif": list(style["font_serif"]),
        "mathtext.fontset": "cm",
        "font.size": style["font_size"],
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
    }


def panels(data: dict) -> list[dict]:
    """One entry per rule, in manuscript order: the nine sampled, then the three exact.

    ``exponents`` holds every exponent that is not ``-inf``. For the sampled
    rules the spectrum of each sample is sorted descending, so the ``-inf`` ones
    are the last ``N - rank`` of it and are dropped; dropping by rank rather
    than by value matters, because for rules 26 and 122 the floating-point QR
    returns a large finite number for directions that are in fact annihilated.
    The affine spectra are censored the same way, which removes exactly the
    rounding-level values the closed form produces where a singular value is
    algebraically zero.
    """
    N = int(data["N"])
    out = []
    for i, rule in enumerate(data["rules"]):
        spectra, ranks = data["spectra"][i], data["ranks"][i]
        out.append({
            "rule": int(rule),
            "exponents": np.concatenate([s[:r] for s, r in zip(spectra, ranks)]),
            "mle": float(spectra[:, 0].mean()),
            "mle_is_a_mean": True,
            "mle_spread": percentile_offsets(spectra[:, 0]),
            "minus_inf_share": float(100.0 * (N - ranks).mean() / N),
        })
    for i, rule in enumerate(data.get("affine_rules", [])):
        spectrum, rank = data["affine_spectra"][i], int(data["affine_ranks"][i])
        out.append({
            "rule": int(rule),
            "exponents": spectrum[:rank],
            "mle": float(spectrum[0]),
            "mle_is_a_mean": False,
            "mle_spread": None,        # the closed form has no sampling spread
            "minus_inf_share": 100.0 * (N - rank) / N,
        })
    return out


def percentile_offsets(values, low: float = 16.0, high: float = 84.0) -> tuple:
    """The 16th and 84th percentiles of a sample, as offsets from its mean.

    The asymmetric counterpart of one standard deviation, and asymmetric it
    is: over the 40 samples rule 73 runs from -0.010 to +0.047 about a mean of
    0.915, because a configuration that keeps more of the tangent space alive
    also stretches faster. A standard deviation would hide that.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [low, high])
    return float(lo - values.mean()), float(hi - values.mean())


def mle_label(panel: dict) -> str:
    """The first annotated line: an exact logarithm, or a mean with its spread.

    Rules 60, 90 and 150 have maximal exponents that are exactly ln 2, ln 2
    and ln 3, so the figure says so rather than rounding them to three
    decimals. The other nine carry the spread of their 40 samples.
    """
    if not panel["mle_is_a_mean"]:
        base = int(round(float(np.exp(panel["mle"]))))
        if abs(panel["mle"] - np.log(base)) < 1e-12:
            return rf"MLE $= \ln\,({base})$"
        return f"MLE $= {panel['mle']:.3f}$"
    low, high = panel["mle_spread"]
    # "MLE" unqualified: the offsets say it is a sample, so "mean" is noise.
    return (f"MLE $= {panel['mle']:.3f}"
            f"^{{{high:+.3f}}}_{{{low:+.3f}}}$")


def outside_range(panel: dict, span) -> tuple:
    """How many of a panel's exponents fall outside the plotted range, and what share."""
    values = panel["exponents"]
    outside = int(((values < span[0]) | (values > span[1])).sum())
    return outside, 100.0 * outside / values.size


def build_figure(data: dict, style: dict | None = None, use_tex: bool = False):
    """Draw the twelve panels. Returns ``(fig, axes)`` with axes in rule order."""
    style = {**STYLE, **(style or {})}
    _style.setup_style(use_tex)
    entries = panels(data)
    if style["bin_range"] is None:
        lo = min(float(p["exponents"].min()) for p in entries)
        hi = max(float(p["exponents"].max()) for p in entries)
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = style["bin_range"]
    edges = np.linspace(lo, hi, style["bins"] + 1)
    # A single exponent in the most populous panel is the smallest visible bar.
    ylim = style["ylim"] or (0.5 / max(p["exponents"].size for p in entries), 1.0)
    rows = -(-len(entries) // 2)

    with mpl.rc_context(_rc(style)):
        fig, axes = plt.subplots(rows, 2, figsize=style["figsize"], sharex=True,
                                 sharey=True,
                                 gridspec_kw={"hspace": style["hspace"],
                                              "wspace": style["wspace"]})
        for entry, ax in zip(entries, axes.flat):
            values = entry["exponents"]
            ax.hist(values, bins=edges, weights=np.full(values.size, 1.0 / values.size),
                    color=style["bar_colour"], edgecolor=style["bar_edge"],
                    linewidth=style["bar_edge_width"])
            ax.axvline(entry["mle"], color=style["mle_colour"], lw=style["mle_width"],
                       ls=style["mle_style"])
            ax.set_yscale("log")
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(*ylim)
            ax.set_xticks(list(style["xticks"]))
            if style["yticks"] is not None:
                ax.set_yticks(list(style["yticks"]))
            ax.yaxis.set_minor_locator(NullLocator())     # major decades only
            ax.set_title(f"Rule {entry['rule']}", loc="left")
            ax.annotate(f"{mle_label(entry)}\n"
                        f"${entry['minus_inf_share']:.1f}\\,\\%$ at $-\\infty$",
                        xy=(0.03, 0.92), xycoords="axes fraction", ha="left", va="top",
                        fontsize=style["annotation_size"])
        for ax in axes.flat[len(entries):]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\Lambda_k$")
        fig.supylabel("Relative frequency", x=style["ylabel_x"],
                      fontsize=style["label_size"])
    return fig, axes


def load_or_compute(cache: Path, recompute: bool = False) -> dict:
    """Read the cached run; recompute it only when explicitly asked.

    The cache is committed, so the normal path is a read. Regenerating it is a
    nine-core-hour run, which ``reproduce.py all`` must never trigger by
    accident, so a missing cache is an error with instructions rather than a
    silent rerun.
    """
    if recompute:
        print("running data/make_nonaffine_spectra.py (hours, see its docstring)",
              flush=True)
        proc = subprocess.run([sys.executable, "data/make_nonaffine_spectra.py",
                               "--recompute"], cwd=REPO_ROOT)
        if proc.returncode != 0:
            raise SystemExit("the computation failed; see the output above")
    elif not cache.exists():
        raise SystemExit(
            f"{cache} is missing. It is committed with the repository; if it "
            "has been removed, regenerate it with\n"
            "    python data/make_nonaffine_spectra.py --recompute\n"
            "which takes a few hours, or pass --recompute to this script.")
    return run.load(cache)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=str(run.CACHE))
    p.add_argument("--bins", type=int, default=STYLE["bins"])
    p.add_argument("--recompute", action="store_true",
                   help="redo the multi-hour computation before drawing")
    p.add_argument("--output", default=None)
    p.add_argument("--tex", action="store_true",
                   help="render text with LaTeX (needs a TeX installation; default: mathtext)")
    args = p.parse_args(argv)

    data = load_or_compute(Path(args.cache), args.recompute)
    span = STYLE["bin_range"]
    print(f"N = {int(data['N'])}, T = {int(data['T'])}, burn-in {int(data['burn'])}, "
          f"{int(data['samples'])} samples per non-affine rule; "
          "the affine rules are the closed form")
    for entry in panels(data):
        values = entry["exponents"]
        outside, share = outside_range(entry, span)
        label = "MLE"
        spread = ("" if entry["mle_spread"] is None else
                  " [{:+.4f} {:+.4f}]".format(*entry["mle_spread"][::-1]))
        print(f"  rule {entry['rule']:4d}: {label} {entry['mle']:.4f}{spread:18s}, "
              f"{entry['minus_inf_share']:5.2f} % at -inf, {values.size:6d} exponents, "
              f"range {values.min():7.3f} to {values.max():6.3f}, "
              f"{outside:5d} outside the axes ({share:.2f} %)")
    fig, _ = build_figure(data, style={"bins": args.bins}, use_tex=args.tex)
    path = _style.save(fig, "lyapunov_spectra_nonaffine_ecas", args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
