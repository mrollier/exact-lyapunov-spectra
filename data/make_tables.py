#!/usr/bin/env python
"""Generate the manuscript tables as CSV files.

* Table 1 (T1): the 16 affine ECAs with their gradients and gradient weights.
* Table 2 (T2): the structure factor K(k, l) and self-inclusive parity MLE for
  the three 2-D neighbourhoods.

(The corrected 88-rule gradient table -- Table 3 in the manuscript numbering used
by the brief -- is produced separately by ``verify_vichniac.py``.)

All values are computed from the verified core, not transcribed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lyapunov.rules import affine_ecas, affine_gradient, gradient_weight
from lyapunov.spectra import (
    parity_2d_mle,
    VON_NEUMANN_2D,
    MOORE_2D,
    VON_NEUMANN_R2_2D,
)

TABLE_DIR = Path(__file__).resolve().parent / "tables"


def affine_eca_table() -> list[dict]:
    """T1: one row per affine ECA (grouped naturally by gradient weight)."""
    rows = []
    seen = set()
    for rule in sorted(affine_ecas(), key=lambda r: (gradient_weight(r), r)):
        if rule in seen:
            continue
        complement = 255 - rule
        seen.update({rule, complement})
        g = affine_gradient(rule)
        rows.append({
            "rule": rule,
            "complement": complement,
            "gradient_a_minus": g[0],
            "gradient_a_centre": g[1],
            "gradient_a_plus": g[2],
            "gradient_weight": sum(g),
        })
    return rows


def structure_factor_table() -> list[dict]:
    """T2: structure factor description and parity MLE per 2-D neighbourhood."""
    return [
        {
            "neighbourhood": "von Neumann",
            "n_neighbours": len(VON_NEUMANN_2D),
            "structure_factor_K": "2 cos(a) + 2 cos(b)",
            "parity_mle": f"ln(5) = {parity_2d_mle(VON_NEUMANN_2D):.6f}",
        },
        {
            "neighbourhood": "Moore",
            "n_neighbours": len(MOORE_2D),
            "structure_factor_K": "2 cos(a) + 2 cos(b) + 4 cos(a) cos(b)",
            "parity_mle": f"ln(9) = {parity_2d_mle(MOORE_2D):.6f}",
        },
        {
            "neighbourhood": "radius-2 von Neumann",
            "n_neighbours": len(VON_NEUMANN_R2_2D),
            "structure_factor_K": ("2 cos(a) + 2 cos(b) + 4 cos(a) cos(b) "
                                   "+ 2 cos(2a) + 2 cos(2b)"),
            "parity_mle": f"ln(13) = {parity_2d_mle(VON_NEUMANN_R2_2D):.6f}",
        },
    ]


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    t1 = affine_eca_table()
    t2 = structure_factor_table()
    _write_csv(t1, TABLE_DIR / "affine_ecas.csv")
    _write_csv(t2, TABLE_DIR / "structure_factors.csv")
    print(f"Wrote {len(t1)} affine-ECA rows to {TABLE_DIR / 'affine_ecas.csv'}")
    print(f"Wrote {len(t2)} neighbourhood rows to {TABLE_DIR / 'structure_factors.csv'}")


if __name__ == "__main__":
    main()
