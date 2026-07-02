"""Shared plotting style and helpers for the figure scripts.

Kept deliberately small: a consistent Matplotlib style, a two-colour map for the
binary defect patterns, and a save helper that writes to ``output/``. Using
Matplotlib's mathtext (rather than a full LaTeX installation) keeps the figures
reproducible on a clean machine; pass ``use_tex=True`` only if a TeX system is
available.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Repository root and the output directory that all figures write to.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

# Colours: a pale background and a dark red for defects (readable in print).
DEFECT_BG = "#fbecec"
DEFECT_FG = "#8b0000"
LINE_BLUE = "#1f5fb4"
LINE_BLACK = "#000000"
ACCENT_RED = "#c02020"


def setup_style(use_tex: bool = False) -> None:
    """Apply the shared Matplotlib rcParams."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": False,
        "text.usetex": bool(use_tex),
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
    })


def defect_cmap() -> ListedColormap:
    """Two-colour map: pale background (0) and dark red defect (1)."""
    return ListedColormap([DEFECT_BG, DEFECT_FG])


def save(fig, stem: str, output: str | None = None, formats=("pdf",)) -> Path:
    """Save ``fig`` to ``output`` (or ``output/<stem>.<fmt>``) and return the path."""
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    last = None
    for fmt in formats:
        last = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(last, bbox_inches="tight")
    plt.close(fig)
    return last
