# Orientation for future Claude Code sessions

This repository is the reproducibility package for the article *"Exact Lyapunov
spectra of affine cellular automata and the parity rule on networks"* (Rollier &
Baetens, submitted to *Chaos, Solitons & Fractals*). Every figure and every
quantitative claim in the paper is reproducible here with one command.

## Layout
- `src/lyapunov/` — the verified maths core. Figures and tests import from here;
  the maths is never duplicated across scripts.
- `figures/` — one standalone script per manuscript figure, writing to `output/`.
- `verification/` — pytest checks, one per paper claim (C1–C7) plus core unit
  tests. `pytest` must exit zero.
- `data/` — deterministic (seeded) regeneration of graphs and the CSV tables.
- `docs/provenance.md` — figure/claim → script → command → expected → status.
- `reproduce.py all` regenerates everything; `reproduce.py quick` is the CI subset.

## The one hazard to remember
`numpy.linalg.matrix_power` **overflows silently** for moderate exponents,
corrupting A^t and Jacobian-power computations. All exact powers go through
`src/lyapunov/gf2.py` (`gf2_matrix_power` for mod-2 defect patterns,
`int_matrix_power` for exact integer walk counts via Python-object dtype).

Precise statement (verified in `verification/test_c7_numerical_artefact.py`):
- **Magnitudes**: int64 `matrix_power` wraps to garbage/negative counts.
- **Parity via float**: computing A^t in float64 then `% 2` gives the wrong
  defect pattern (float64 loses the low bit above 2^53). This is the realistic bug.
- **Parity via pure int64**: `matrix_power(int64) % 2` *accidentally* keeps the
  right parity (two's-complement wraps mod 2^64, and 2 | 2^64) — do not rely on
  this coincidence; use `gf2_matrix_power`, which is correct in every case.

Never reintroduce `np.linalg.matrix_power` on a matrix whose true power you need.

## Conventions
- UK English in prose and comments (behaviour, artefact, organise, neighbour).
- Determinism: every RNG seed is fixed and documented in the calling script and
  in `data/make_graphs.py`. Two runs give identical numbers.
- Development is test-driven: tests encode the paper's exact analytical values
  (e.g. MLE of rule 150 is exactly ln 3), written before the implementation.

## Scope note
This package covers the submitted CSF paper only (5 figures, 3 tables, claims
C1–C7). The separate network-automata-robustness project (LLNA training, FSSP,
impact analysis) is deliberately not included.
