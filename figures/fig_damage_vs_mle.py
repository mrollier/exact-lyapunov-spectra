#!/usr/bin/env python
"""Figure 7 -- Boolean damage against the maximal Lyapunov exponent, every rule.

Stem: damage_vs_mle

Three scatter plots, one above the other with a shared x axis. Top: the 88
ECAs up to reflection and conjugation on a ring; middle: the 528
outer-totalistic von Neumann rules up to conjugation on a torus; bottom: 2000
of the 131 328 outer-totalistic Moore rules up to conjugation, drawn uniformly
from the classes (the family is too large to enumerate). Each dot is one rule: the
mean relative damage ``D_norm`` caused by a single flipped cell (damaged
cells over the cells in the maximal light cone, averaged over the final steps
and over the initial configurations) against the mean maximal exponent of the
Boolean Jacobian on the same trajectories. Rules whose tangent vector is
annihilated exactly have exponent -inf and sit in a marked column at the left
of the axis; they take no part in
Spearman's rho. Dashed lines mark the exact exponents of the affine and
parity rules (ln(2), ln(3); ln(4), ln(5); ln(8), ln(9)). For the sampled Moore
family the script also prints a seeded bootstrap interval for rho.

The numbers come from the committed caches written by
``data/make_damage_mle.py`` (see its docstring for the parameters); the
script only draws. The caches also hold the front speed and the cone fill,
which the tables report but this figure does not show.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _style

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from make_damage_mle import cache_path, load  # noqa: E402

# A few ECAs named in the text; the 2-D rules are not annotated.
LABELS_1D = {30: (3, 3), 90: (3, -8), 110: (-3, 3), 150: (-3, -8)}
PANELS = ("1d", "2d", "moore")
REFERENCE_LINES = {"1d": ((np.log(2), "ln(2)"), (np.log(3), "ln(3)")),
                   "2d": ((np.log(4), "ln(4)"), (np.log(5), "ln(5)")),
                   "moore": ((np.log(8), "ln(8)", "left"), (np.log(9), "ln(9)"))}
TITLES = {"1d": "88 ECAs on a ring",
          "2d": "528 outer-totalistic von Neumann rules on a torus",
          "moore": "2000 sampled outer-totalistic Moore rules on a torus"}
# Height (axes fraction) of the reference-line labels: the top of the panel,
# except where the top right is full of points.
LABEL_HEIGHT = {"1d": 0.96, "2d": 0.96, "moore": 0.56}
BOOTSTRAP = 2000                          # resamples of the rules for the Moore interval
GREY = "#9a9a9a"
# Serif text with Times-like maths, larger than the shared style, for a
# narrow two-row figure that is read at column width. One size for every
# piece of text except the tick labels, so the panels look alike.
TEXT = 13
TICK = 12
FONT = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": TEXT, "axes.titlesize": TEXT, "axes.labelsize": TEXT,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
}


def per_rule(data: dict) -> dict:
    """Per-rule means from the per-sample cache; -inf where every sample died."""
    mle = data["mle"]
    finite = np.isfinite(mle)
    with np.errstate(invalid="ignore"):
        mle_mean = np.where(finite.any(axis=1),
                            np.nansum(np.where(finite, mle, np.nan), axis=1) / finite.sum(axis=1),
                            -np.inf)
    return {"rules": data["rules"], "mle": mle_mean, "D_norm": data["D_norm"].mean(axis=1),
            "n_inf": (~finite).sum(axis=1)}


def rho(x, y) -> float:
    """Spearman's rho over the rules with a finite exponent."""
    keep = np.isfinite(x)
    return float(spearmanr(x[keep], y[keep]).statistic)


def rho_interval(x, y, seed: int = 0, n: int = BOOTSTRAP) -> tuple:
    """Seeded percentile bootstrap (2.5 %, 97.5 %) of rho over the sampled rules."""
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    rng = np.random.default_rng(seed)
    draws = [spearmanr(x[idx], y[idx]).statistic
             for idx in rng.integers(0, x.size, size=(n, x.size))]
    return tuple(float(q) for q in np.percentile(draws, [2.5, 97.5]))


def draw_panel(ax, summary, panel, x_max, x_inf):
    x, y = summary["mle"], summary["D_norm"]
    finite = np.isfinite(x)
    xs = np.where(finite, x, x_inf)             # the -inf rules go in their own column
    # The -inf rules are ordinary data points, drawn like the rest.
    ax.scatter(xs, y, s=14, color=_style.LINE_BLUE, edgecolors="white",
               linewidths=0.4, zorder=3)
    ax.axvline(x_inf / 2, color=GREY, lw=0.6, ls=":", zorder=1)
    for value, name, *side in REFERENCE_LINES[panel]:
        # Labels sit to the right of their line unless the lines are too close
        # (ln 8 and ln 9), in which case the first goes to the left.
        left = side == ["left"]
        ax.axvline(value, color=_style.LINE_BLACK, lw=0.8, ls="--", zorder=1)
        ax.annotate(name, (value, LABEL_HEIGHT[panel]), xytext=(-2 if left else 2, 0),
                    textcoords="offset points",
                    xycoords=ax.get_xaxis_transform(), ha="right" if left else "left",
                    va="top", rotation=90)
    if panel == "1d":
        for i, rule in enumerate(summary["rules"]):
            if int(rule) in LABELS_1D:
                dx, dy = LABELS_1D[int(rule)]
                ax.annotate(str(rule), (xs[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                            ha="left" if dx > 0 else "right")
    ax.text(0.03, 0.95, f"Spearman $\\rho$ = {rho(x, y):.2f}", transform=ax.transAxes,
            va="top")
    ax.set_title(TITLES[panel], loc="left")
    ax.set_ylim(-0.02, 0.55)


def build_figure(caches: dict, use_tex: bool = False):
    _style.setup_style(use_tex)
    plt.rcParams.update(FONT)
    summaries = {panel: per_rule(caches[panel]) for panel in PANELS}
    # One x axis for all rows, so the -inf column and the limits are shared.
    x_max = max(s["mle"][np.isfinite(s["mle"])].max() for s in summaries.values())
    x_inf = -0.16 * x_max
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(4.6, 6.6), sharex=True)
    for ax, panel in zip(axes, PANELS):
        draw_panel(ax, summaries[panel], panel, x_max, x_inf)
    ax = axes[-1]
    ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    ax.set_xticks([x_inf] + ticks)
    ax.set_xticklabels([r"$-\infty$"] + [f"{t:g}" for t in ticks])
    ax.set_xlim(x_inf - 0.07 * x_max, x_max * 1.13)   # room for the ln(9) label
    ax.set_xlabel("Mean MLE")
    fig.tight_layout(h_pad=0.9)
    # One y label for all rows, on the middle axes and centred on the stack.
    # Only its height is set: matplotlib keeps placing it just left of the
    # tick labels, so it sits as close to the axis as an ordinary label would.
    host = axes[len(axes) // 2]
    for _ in range(2):                          # once more after the label takes its space
        top, bottom, mid = axes[0].get_position(), axes[-1].get_position(), host.get_position()
        y_mid = ((bottom.y0 + top.y1) / 2 - mid.y0) / mid.height
        host.set_ylabel("mean relative damage", y=y_mid)
        fig.tight_layout(h_pad=0.9)
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-1d", default=str(cache_path(1)))
    p.add_argument("--cache-2d", default=str(cache_path(2)))
    p.add_argument("--cache-moore", default=str(cache_path(2, "moore")))
    p.add_argument("--output", default=None, help="Output path (default output/<stem>.pdf)")
    p.add_argument("--no-tex", action="store_true", help="Use mathtext, not LaTeX (default).")
    args = p.parse_args(argv)
    caches = {}
    commands = {"1d": "--dim 1", "2d": "--dim 2", "moore": "--dim 2 --neighbourhood moore"}
    for panel, path in (("1d", args.cache_1d), ("2d", args.cache_2d), ("moore", args.cache_moore)):
        if not Path(path).exists():
            print(f"missing {path}: run  python data/make_damage_mle.py {commands[panel]}",
                  file=sys.stderr)
            return 1
        caches[panel] = load(Path(path))
    fig = build_figure(caches, use_tex=False)
    path = _style.save(fig, "damage_vs_mle", args.output)
    print(f"Wrote {path}")
    for panel in PANELS:
        s = per_rule(caches[panel])
        line = (f"  {panel}: Spearman rho(Lambda_max, D_norm) = {rho(s['mle'], s['D_norm']):.3f}; "
                f"{int((~np.isfinite(s['mle'])).sum())} rules at -inf")
        if panel == "moore":
            lo, hi = rho_interval(s["mle"], s["D_norm"])
            line += f"; bootstrap 95 % interval [{lo:.3f}, {hi:.3f}] over the sampled classes"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
