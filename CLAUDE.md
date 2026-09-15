# Orientation for future Claude Code sessions

This repository is the reproducibility package for the article *"Exact Lyapunov
spectra of affine cellular automata and the parity rule on networks"* (Rollier &
Baetens, *Chaos, Solitons & Fractals*, revised version resubmitted 14 September
2026: 6 figures, 4 tables, appendices A-C). Every figure and every quantitative
claim in the paper is reproducible here with one command. Figure and table
numbers in the repository are those of the revised manuscript.

## Layout
- `src/lyapunov/` — the verified maths core. Figures and tests import from here;
  the maths is never duplicated across scripts.
- `figures/` — one standalone script per manuscript figure, writing to `output/`.
- `verification/` — pytest checks of the paper's claims (C1–C12; C8 is out of
  scope, C9 lives in the two Benettin test files) plus core unit tests.
  `pytest` must exit zero; `conftest.py` puts `scripts/` on the path.
- `scripts/` — the generators: `verify_vichniac.py` (Tab. 2, App. A),
  `make_tables.py`, `make_graphs.py`, the two cache builders and
  `bench_workers.py`; deterministic (seeded), all writing into `data/`. The
  root `verify_vichniac.py` is only a launcher, kept because App. A prints
  that command.
- `data/` — data only: the CSV tables and the committed caches.
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
  in `scripts/make_graphs.py`. Two runs give identical numbers.
- Development is test-driven: tests encode the paper's exact analytical values
  (e.g. MLE of rule 150 is exactly ln 3), written before the implementation.

## Scope note
This package covers the revised CSF paper: Figs. 1-4 and Tabs. 2-4 of the
original submission (claims C1-C7, C9), Fig. 5 (Boolean damage against the
maximal exponent for all 88 ECAs, all 528 outer-totalistic von Neumann rules
and 2000 sampled outer-totalistic Moore rules, App. B, C11-C12), Fig. 6
(corrected non-affine spectra, App. C, C10) and the 88-rule spectra catalogue
App. C cites. The defect-propagation-on-networks figure of the original
submission is kept as supplementary figure S1 (`fig_defect_topologies.py`,
eigenvector centrality in `parity.py`); the revised manuscript makes no claim
about it. The separate network-automata-robustness project (LLNA training,
FSSP, impact analysis) is deliberately not included.
