#!/usr/bin/env python
"""Regenerate the graph topologies used in the defect-propagation figure (Fig. 5).

Every graph is built from a fixed, documented seed, so two runs produce byte-for
-byte identical adjacency matrices. The matrices are cached as ``.npz`` files in
``data/graphs/``; the figure script loads them (and regenerates them here if they
are missing).

Topologies (adjacency matrices, undirected, unweighted):
* ring   : cycle graph C_N (N=225)              (rule-90 Sierpinski defect)
* grid   : periodic 2-D lattice, Moore neighbourhood, 15 x 15 torus (225 nodes)
* ws     : Watts-Strogatz small-world, N=225, mean degree 6, rewiring p=0.2
* ba     : Barabasi-Albert scale-free,  N=225, m=3

Note on sizes: all four topologies use 225 nodes (ring N=225; grid a 15 x 15
torus; WS and BA N=225) so the figure panels are directly comparable. WS keeps
the manuscript's mean degree 6 and rewiring p=0.2; BA keeps m=3. These choices
are recorded in SEEDS below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import networkx as nx

GRAPH_DIR = Path(__file__).resolve().parent / "graphs"

# Fixed generation parameters and seeds (change nothing here to reproduce).
# All four topologies use 225 nodes (grid = 15 x 15) so the figure panels are
# directly comparable. WS keeps mean degree 6 and p=0.2; BA keeps m=3.
SEEDS = {
    "ws": dict(N=225, k=6, p=0.2, seed=20240601),
    "ba": dict(N=225, m=3, seed=20240601),
    "ring": dict(N=225),
    "grid": dict(L=15, moore=True),
}


def _moore_torus_adjacency(L: int) -> np.ndarray:
    """Adjacency of an L x L torus with the 8-neighbour Moore neighbourhood."""
    N = L * L
    A = np.zeros((N, N), dtype=int)
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    for x in range(L):
        for y in range(L):
            i = x * L + y
            for dx, dy in offsets:
                j = ((x + dx) % L) * L + ((y + dy) % L)
                A[i, j] = 1
    return A


def build_graphs() -> dict[str, np.ndarray]:
    """Return the four adjacency matrices, built deterministically from SEEDS."""
    ws = nx.watts_strogatz_graph(SEEDS["ws"]["N"], SEEDS["ws"]["k"],
                                 SEEDS["ws"]["p"], seed=SEEDS["ws"]["seed"])
    ba = nx.barabasi_albert_graph(SEEDS["ba"]["N"], SEEDS["ba"]["m"],
                                  seed=SEEDS["ba"]["seed"])
    ring = nx.cycle_graph(SEEDS["ring"]["N"])
    grid = _moore_torus_adjacency(SEEDS["grid"]["L"])
    return {
        "ring": nx.to_numpy_array(ring, dtype=int),
        "grid": grid,
        "ws": nx.to_numpy_array(ws, dtype=int),
        "ba": nx.to_numpy_array(ba, dtype=int),
    }


def save_graphs(directory: Path = GRAPH_DIR) -> Path:
    """Build and cache all graphs to ``directory`` as ``<name>.npz``."""
    directory.mkdir(parents=True, exist_ok=True)
    graphs = build_graphs()
    for name, A in graphs.items():
        np.savez_compressed(directory / f"{name}.npz", adjacency=A)
    return directory


def load_graph(name: str, directory: Path = GRAPH_DIR) -> np.ndarray:
    """Load a cached adjacency matrix, regenerating the cache if absent."""
    path = directory / f"{name}.npz"
    if not path.exists():
        save_graphs(directory)
    return np.load(path)["adjacency"]


if __name__ == "__main__":
    out = save_graphs()
    for name, A in build_graphs().items():
        print(f"{name:5s}: N={A.shape[0]:4d}  edges={int(A.sum() // 2):5d}")
    print(f"Saved adjacency matrices to {out}")
