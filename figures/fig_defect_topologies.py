#!/usr/bin/env python
"""Manuscript Figure 5 -- defect propagation across topologies.

Stem: defect_propagation_networks_parity

The true Boolean difference pattern of the (self-exclusive) parity rule started
from a single defect, Delta s^t = A^t e_j (mod 2), on four topologies: a ring, a
periodic 2-D lattice (Moore), a Watts-Strogatz small-world graph and a
Barabasi-Albert scale-free graph. Time runs downward; a dark-red cell marks a
node whose state differs between the two configurations. Nodes (columns) are
ordered by eigenvector centrality, as in the manuscript.

The ring gives the Sierpinski triangle (binomial coefficients modulo two); the
grid a periodic wavefront; the small-world and scale-free graphs lose that
regularity because rewired edges and hubs carry the defect to distant nodes.

Graphs are the seeded topologies of data/make_graphs.py (deterministic).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lyapunov.parity import defect_pattern, eigenvector_centrality
import _style

# Make data/make_graphs.py importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from make_graphs import load_graph  # noqa: E402

PANELS = [("ring", "Ring"), ("grid", "Grid"), ("ws", "WS"), ("ba", "BA")]


def _seed_node(name: str, A: np.ndarray) -> int:
    N = A.shape[0]
    if name == "ring":
        return N // 2
    if name == "grid":
        L = int(round(np.sqrt(N)))
        return (L // 2) * L + L // 2
    # small-world / scale-free: seed at the highest-centrality node (a hub)
    return int(np.argmax(eigenvector_centrality(A)))


def build_figure(use_tex: bool = False):
    _style.setup_style(use_tex)
    cmap = _style.defect_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for ax, (name, label) in zip(axes.ravel(), PANELS):
        A = load_graph(name)
        N = A.shape[0]
        T = N // 2
        seed = _seed_node(name, A)
        pattern = defect_pattern(A, seed=seed, T=T, self_inclusive=False)
        order = np.argsort(eigenvector_centrality(A), kind="stable")
        ax.imshow(pattern[:, order], cmap=cmap, interpolation="nearest",
                  aspect="auto", vmin=0, vmax=1, origin="upper")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r"Time $\rightarrow$")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=None)
    p.add_argument("--no-tex", action="store_true")
    args = p.parse_args(argv)
    fig = build_figure(use_tex=False)
    path = _style.save(fig, "defect_propagation_networks_parity", args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
