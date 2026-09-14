#!/usr/bin/env python
"""Revised manuscript Figure 3 -- the affine spectrum as a benchmark, with the
per-exponent error of Benettin's algorithm.

Stem: convergence_rule150

Panel A is the published Figure 3 (rule 150, N = 101, T = 200): the exact closed
form, Benettin's QR algorithm and direct multiplication in float16 and float64,
with the inset over k <= zoom-k marked on the main axes by its connectors.

Panel B is new: |Lambda_k^est - Lambda_k^exact| against the sorted index k for
two Benettin estimates with the same budget of T steps,
  * no burn-in, running average over all T steps (the estimate in panel A, which
    carries its initial frame-alignment transient diluted as 1/T), and
  * a burn-in of B steps followed by the average over the remaining T - B steps.
Both are slices of one stored run of per-step log stretches
(``lyapunov.benettin.benettin_log_stretch``); sorting before comparison is
essential because nearly degenerate pairs exchange places while converging.
Panel B is drawn in black alone, so that colour means method in panel A only.

The script also prints (and asserts) the reference table for a window of
W = 1000 steps after burn-ins B = 0, 100, 300, 1000, 3000 and for the running
average at T = 4000, plus the smallest burn-in on the grid that brings the
pairwise envelope max(e_k, e_{k+1}) below 1e-2 and 1e-3 for each k.

``notebooks/04_convergence_figure.ipynb`` drives the same functions
interactively; every adjustable quantity is in the ``STYLE`` dictionary below
and can be overridden per call.

Usage:
    python figures/make_convergence_figure.py --rule 150 --N 101 --T 200 --burn 100 --zoom-k 35
"""
from __future__ import annotations

import argparse
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # single-thread QR: faster here, and run-to-run identical

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from lyapunov.jacobian import eca_jacobian
from lyapunov.benettin import (
    closed_form_spectrum,
    benettin_log_stretch,
    windowed_spectrum,
    cumulative_spectrum,
)
import _style
from make_benchmark_figure import _spectra          # the four spectra of the published figure

STEPS = 4000                       # one QR run serves every estimator below
W_REF = 1000                       # window of the reference table
B_REF = (0, 100, 300, 1000, 3000)
B_GRID = (0, 30, 100, 300, 1000, 3000)
REFERENCE = {                      # (fraction within 1e-3, within 1e-2, max error)
    "running average, T = 4000": (0.09, 1.00, 6.9e-3),
    0: (0.02, 0.45, 2.8e-2), 100: (0.79, 0.97, 1.8e-2), 300: (0.89, 0.98, 1.8e-2),
    1000: (1.00, 1.00, 6.5e-5), 3000: (1.00, 1.00, 4.0e-7),
}
EPS_FLOOR = 1e-15                  # plotted error is clipped here (machine precision)

# Everything adjustable about the drawing. Pass a dict of overrides as ``style``
# to :func:`build_figure`; missing keys keep the values below.
STYLE = {
    "figsize": (5.9, 6.4),          # narrower than the published figure, same panel-A proportions
    "height_ratios": (1.55, 1.0),
    "hspace": 0.10,
    "font_size": 11,                # base size; labels and ticks are set relative to it
    "label_size": 12,
    "ylabel_size": 14,              # the y labels carry the expressions, so a size of their own
    "legend_size": 10,
    "tick_size": 10,
    # Serif text to match the manuscript. Computer Modern first, Times as the
    # fallback; maths is Computer Modern either way (mathtext.fontset = "cm").
    # Passing use_tex=True to build_figure switches to real LaTeX typesetting.
    "font_family": "serif",
    "font_serif": ("CMU Serif", "Times New Roman", "DejaVu Serif"),
    "inset_tick_size": 8,
    "inset_title_size": 10,
    "panel_label_size": 12,
    "inset_rect": (0.10, 0.13, 0.45, 0.42),     # x0, y0, w, h in axes coordinates
    "inset_ylim": (0.58, 1.16),                 # data limits, so the marked box is exact
    "inset_xticks": (1, 5, 10, 15, 20, 25, 30, 35),
    "inset_yticks": (0.6, 0.7, 0.8, 0.9, 1.0, 1.1),
    "xticks": (1, 25, 50, 75, 101),
    "marker_size": 4.5,
    "line_width": 1.4,
    "connector_colour": "0.55",
    "connector_width": 0.7,
    "legend_bbox": (0.5, 1.005),    # legend above panel A, two columns, no frame
    "panel_label_xy": (-0.175, 1.0),
    "error_ylim": (EPS_FLOOR / 3, 3.0),
    "error_yticks": (1e0, 1e-4, 1e-8, 1e-12),   # every fourth decade: the log axis stays legible
}


def _rc(style):
    """rcParams overriding the repo defaults for this figure only."""
    return {
        "font.size": style["font_size"],
        "font.family": style["font_family"],
        "font.serif": list(style["font_serif"]),
        "mathtext.fontset": "cm",
        "axes.labelsize": style["label_size"],
        "axes.titlesize": style["label_size"],
        "legend.fontsize": style["legend_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
    }


def panel_spectrum(ax, spec, k, style, markers: bool = True):
    """Panel A: the four spectra, styled as in the published Figure 3."""
    ms = style["marker_size"] if markers else style["marker_size"] - 1.3
    ax.plot(k, spec["closed"], "-", color=_style.LINE_BLACK, lw=style["line_width"],
            label="Exact (closed form)")
    ax.plot(k, spec["benettin"], "o", mfc="none", mec=_style.LINE_BLUE, ms=ms, mew=0.9,
            label="Benettin")
    ax.plot(k, spec["f16"], "^", color=_style.ACCENT_RED, ms=ms - 0.7, lw=0,
            label="Direct mult. (float16)")
    ax.plot(k, spec["f64"], "x", color=_style.ACCENT_RED, ms=ms - 0.7, mew=0.9, lw=0,
            label="Direct mult. (float64)")


def panel_error(ax, k, err_run, err_win, T, burn, style):
    """Panel B: the two same-budget Benettin errors, in black alone."""
    ms = style["marker_size"]
    ax.plot(k, np.maximum(err_run, EPS_FLOOR), "o", mfc="none", mec="black", ms=ms, mew=0.8,
            label=f"no burn-in, average over {T} steps")
    ax.plot(k, np.maximum(err_win, EPS_FLOOR), "s", color="black", ms=ms - 1.3, lw=0,
            label=f"burn-in {burn}, average over next {T - burn} steps")
    ax.set_yscale("log")
    ax.set_ylim(*style["error_ylim"])
    lo, hi = sorted(style["error_ylim"])
    ax.set_yticks([t for t in style["error_yticks"] if lo <= t <= hi])
    ax.set_ylabel(r"$|\Lambda_k^{\rm est} - \Lambda_k|$", fontsize=style["ylabel_size"])
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderaxespad=0.4)


def mark_inset(ax, axins, style):
    """Outline the region the inset shows and join it to the inset by two lines.

    The rectangle is the inset's own data limits, drawn on the parent axes; the
    connectors run from its two lower corners to the inset's two upper corners,
    as in the published figure. ``ConnectionPatch`` mixes the parent's data
    coordinates with its axes coordinates, so nothing has to be transformed by
    hand.
    """
    ax.autoscale_view()
    (x0, x1), (y0, y1) = axins.get_xlim(), axins.get_ylim()
    grey, lw = style["connector_colour"], style["connector_width"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=grey, lw=lw, zorder=4))
    ix0, iy0, iw, ih = style["inset_rect"]
    for x_rect, x_inset in ((x0, ix0), (x1, ix0 + iw)):
        ax.add_artist(ConnectionPatch(
            xyA=(x_rect, y0), coordsA=ax.transData,
            xyB=(x_inset, iy0 + ih), coordsB=ax.transAxes,
            color=grey, lw=lw, zorder=4))


def _summary(err):
    return float(np.mean(err < 1e-3)), float(np.mean(err < 1e-2)), float(err.max())


def compute(rule: int, N: int, T: int, burn: int, steps: int = STEPS):
    """Everything the figure needs: the four spectra, the stored QR run, the two errors."""
    J = eca_jacobian(rule, N)
    exact = closed_form_spectrum(rule, N)
    L = benettin_log_stretch(J, max(steps, T))
    data = {"J": J, "exact": exact, "L": L, "rule": rule, "N": N}
    return _restate(data, rule, N, T, burn)


def _restate(data: dict, rule: int, N: int, T: int, burn: int) -> dict:
    """Put ``data`` at horizon ``T`` and burn-in ``burn``, reusing the stored QR run.

    Only the two error curves and, when ``T`` changes, the four spectra of panel A
    have to be redone; the QR run itself is horizon-independent, so switching
    burn-in costs nothing.
    """
    data = dict(data)
    if data.get("N", N) != N or data["L"].shape[1] != N:
        raise ValueError(f"stored run has N = {data['L'].shape[1]}, not {N}.")
    if data.get("T") != T or "spec" not in data:
        data["spec"] = _spectra(rule, N, T)
    exact, L = data["exact"], data["L"]
    data["T"], data["burn"], data["rule"], data["N"] = T, burn, rule, N
    data["err_run"] = np.abs(cumulative_spectrum(L, T) - exact)                # no burn-in, T steps
    data["err_win"] = np.abs(windowed_spectrum(L, burn, T - burn) - exact)     # burn-in, then T - burn
    return data


def build_figure(rule: int = 150, N: int = 101, T: int = 200, burn: int = 100,
                 zoom_k: int = 35, style: dict | None = None, data: dict | None = None,
                 use_tex: bool = False):
    """Draw the two-panel figure. Returns ``(fig, axes, data)``.

    ``axes`` is a dict with keys ``"A"``, ``"B"`` and ``"inset"``, so the caller
    can adjust limits, ticks or annotations afterwards. ``style`` overrides any
    key of :data:`STYLE`. ``data`` reuses the output of :func:`compute`, which is
    the slow part (one QR run of 4000 steps, about 2 s); its horizon and burn-in
    are restated to ``T`` and ``burn`` without repeating that run.
    """
    st = dict(STYLE, **(style or {}))
    _style.setup_style(use_tex)
    data = compute(rule, N, T, burn) if data is None else _restate(data, rule, N, T, burn)
    exact, spec = data["exact"], data["spec"]
    k = np.arange(1, N + 1)

    with mpl.rc_context(_rc(st)):
        fig, (ax, axe) = plt.subplots(
            2, 1, figsize=st["figsize"], sharex=True,
            gridspec_kw={"height_ratios": list(st["height_ratios"]), "hspace": st["hspace"]})

        panel_spectrum(ax, spec, k, st, markers=True)
        ax.set_ylabel(r"Lyapunov exponent $\Lambda_k$", fontsize=st["ylabel_size"])
        ax.set_xlim(1, N)
        ax.set_xticks(list(st["xticks"]))
        ax.legend(loc="lower center", bbox_to_anchor=st["legend_bbox"], ncol=2,
                  frameon=False, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0)

        # Inset over the leading exponents, with connectors to the region it shows.
        axins = ax.inset_axes(list(st["inset_rect"]))
        panel_spectrum(axins, {key: val[:zoom_k] for key, val in spec.items()},
                       k[:zoom_k], st, markers=False)
        axins.set_xlim(1, zoom_k)
        axins.set_ylim(*st["inset_ylim"])
        axins.set_xticks([t for t in st["inset_xticks"] if t <= zoom_k])
        axins.set_yticks([t for t in st["inset_yticks"] if st["inset_ylim"][0] <= t <= st["inset_ylim"][1]])
        axins.tick_params(labelsize=st["inset_tick_size"])
        axins.set_title(f"detail for $k \\leq {zoom_k}$", fontsize=st["inset_title_size"], pad=3)
        mark_inset(ax, axins, st)

        panel_error(axe, k, data["err_run"], data["err_win"], data["T"], data["burn"], st)
        axe.set_xlabel(r"Exponent index $k$ (sorted)")

        # Panel labels clear of the legend, in the margin left of each y axis.
        for label, target in (("(A)", ax), ("(B)", axe)):
            target.text(*st["panel_label_xy"], label, transform=target.transAxes,
                        ha="left", va="bottom", fontsize=st["panel_label_size"])
        fig.align_ylabels([ax, axe])

    return fig, {"A": ax, "B": axe, "inset": axins}, data


def reference_table(L, exact, verbose: bool = True) -> list[str]:
    """Print the W = 1000 table and the derived burn-in distribution; return failures."""
    N = L.shape[1]
    failures = []
    out = []
    out.append(f"reference table, N = {N}, Q0 = I, W = {W_REF}")
    out.append(f"  {'estimator':28s} {'steps':>5s}  {'<1e-3':>6s}  {'<1e-2':>6s}  {'max err':>8s}")
    rows = [("running average, T = 4000", STEPS, cumulative_spectrum(L, STEPS))]
    rows += [(B, B + W_REF, windowed_spectrum(L, B, W_REF)) for B in B_REF]
    for key, steps, est in rows:
        f3, f2, emax = _summary(np.abs(est - exact))
        r3, r2, rmax = REFERENCE[key]
        ok = abs(f3 - r3) <= 0.02 and abs(f2 - r2) <= 0.02 and rmax / 2 <= emax <= rmax * 2
        name = key if isinstance(key, str) else f"windowed, B = {key}"
        out.append(f"  {name:28s} {steps:5d}  {f3:6.2f}  {f2:6.2f}  {emax:8.1e}  [{'ok' if ok else 'FAIL'}]")
        if not ok:
            failures.append(f"{name}: got ({f3:.2f}, {f2:.2f}, {emax:.1e}), reference ({r3}, {r2}, {rmax})")

    # Derived quantity: per k, the smallest grid burn-in whose pairwise-envelope
    # error is below eps. The error is not monotone in B for every k (nearly
    # degenerate pairs in the interior band swap), so two definitions exist:
    # the first grid B that passes, and the first B from which every larger
    # grid B also passes. Both are printed; the k where they differ are listed.
    env_err = []
    for B in B_GRID:
        e = np.abs(windowed_spectrum(L, B, W_REF) - exact)
        env = np.maximum(e[:-1], e[1:])                            # pair (k, k+1)
        env_err.append(np.maximum(np.r_[env, 0.0], np.r_[0.0, env]))   # worst pair containing k
    env_err = np.array(env_err)                                    # (len(B_GRID), N)
    for eps in (1e-2, 1e-3):
        below = env_err < eps
        first = np.array([B_GRID[np.argmax(c)] if c.any() else -1 for c in below.T])
        safe = np.full(N, -1)
        for i in range(len(B_GRID) - 1, -1, -1):
            safe[np.all(below[i:], axis=0)] = B_GRID[i]
        for name, need in (("first B that passes", first), ("first B from which all larger B pass", safe)):
            vals, counts = np.unique(need[need >= 0], return_counts=True)
            dist = ", ".join(f"B = {int(v)}: {c}" for v, c in zip(vals, counts))
            never = int((need < 0).sum())
            out.append(f"  envelope < {eps:g}, {name:38s}: {dist}" + (f"; never: {never}" if never else ""))
        differ = np.where(first != safe)[0] + 1
        if differ.size:
            out.append(f"    the two definitions differ at k = {differ.tolist()}")
    if verbose:
        print("\n".join(out))
    return failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rule", type=int, default=150)
    p.add_argument("--N", type=int, default=101)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--burn", type=int, default=100)
    p.add_argument("--zoom-k", type=int, default=35, dest="zoom_k")
    p.add_argument("--output", default=None)
    p.add_argument("--no-check", action="store_true", help="skip the reference-table assertions")
    args = p.parse_args(argv)
    fig, _, data = build_figure(args.rule, args.N, args.T, args.burn, args.zoom_k)
    for label, err in ((f"no burn-in, T = {args.T}", data["err_run"]),
                       (f"burn-in {args.burn}, window {args.T - args.burn}", data["err_win"])):
        f3, f2, emax = _summary(err)
        print(f"panel B, {label:28s}: within 1e-3 {f3:.2f}, within 1e-2 {f2:.2f}, max error {emax:.1e}")
    path = _style.save(fig, f"convergence_rule{args.rule}", args.output)
    print(f"Wrote {path}")
    if not args.no_check and (args.rule, args.N) == (150, 101):
        failures = reference_table(data["L"], data["exact"])
        if failures:
            print("reference-table check FAILED:\n  " + "\n  ".join(failures))
            return 1
        print("reference-table check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
