# Exact Lyapunov spectra of affine cellular automata and the parity rule on networks

Reproducibility package for the article *"Exact Lyapunov spectra of affine
cellular automata and the parity rule on networks"* by Michiel Rollier and
Jan M. Baetens (BionamiX, Ghent University), submitted to *Chaos, Solitons &
Fractals*.

Every figure, table and quantitative claim in the paper is reproducible from this
repository with a single command, and the central claims are independently
verified by a test suite.

## What the paper shows

Affine Boolean rules — those whose update is an `XOR` of a subset of the inputs
plus a constant — have a **configuration-independent Boolean Jacobian**. Their
Lyapunov spectrum is therefore *exact*: the logarithms of the singular values of
one constant matrix, with no simulation and no limit. On a periodic lattice that
matrix is (multilevel) circulant, so the spectrum is the discrete Fourier
transform of the gradient stencil; for the parity rule on any graph it is the
adjacency matrix, so the spectrum is the logarithm of the absolute graph
spectrum and the single-site perturbation amplitude scales with eigenvector
centrality.

## Install

Requires Python ≥ 3.10 (tested on 3.11).

```bash
python -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -r requirements.txt
pip install -e .
```

Dependencies are pinned exactly in `requirements.txt`
(numpy, scipy, networkx, matplotlib, pytest). The maths core needs only
numpy + scipy; networkx provides the graph topologies and matplotlib the figures.

## Reproduce everything

```bash
python reproduce.py all      # regenerate all figures + tables, run every check
python reproduce.py quick    # fast subset (tables + checks); used by CI
```

- Figures are written to `output/` (git-ignored; regenerable).
- Tables (CSV) to `data/tables/`; seeded graphs to `data/graphs/`.
- The full check suite runs under `pytest` and exits non-zero on any failure.

Individual pieces:

```bash
python figures/make_benchmark_figure.py --rule 150 --N 101 --T 200 --zoom-k 35
python verify_vichniac.py --output data/tables/eca_gradient_table.csv
python -m pytest -q
```

## Repository layout

```
src/lyapunov/        verified maths core (imported by figures and tests)
  gf2.py             int64-safe GF(2) / integer matrix powers (the central fix)
  rules.py           ECA rule tables, Boolean gradients, affine detection, 88 classes
  jacobian.py        circulant / adjacency Jacobians; pure-numpy ECA evolution
  spectra.py         closed-form DFT singular values; neighbourhood structure factor
  benettin.py        Benettin QR + direct-multiplication reference methods
  parity.py          parity rule on graphs; A^t e_j (mod 2); eigenvector centrality
  quine_mccluskey.py Boolean minimiser for the gradient table
  vichniac.py        recompute the 88-rule gradient table; the four corrections
figures/             one standalone script per manuscript figure -> output/
verification/        one pytest check per claim (C1–C7) + core unit tests
data/                make_graphs.py, make_tables.py, generated tables/graphs
verify_vichniac.py   CLI: gradient checker + Quine–McCluskey + CSV export
reproduce.py         single entry point (all | quick)
docs/provenance.md   figure/claim -> script -> command -> expected -> status
```

## Figure and claim map

| Manuscript object | Script / test |
|---|---|
| Fig 1 defect cones (`persistent_defect_eca_diff`) | `figures/fig_defect_cones.py` |
| Fig 2 affine ECA spectra (`..._NO_CLASSES`) | `figures/fig_eca_spectra.py` |
| Fig 3 benchmark (`benchmark_rule150`) | `figures/make_benchmark_figure.py` |
| Fig 4 2-D parity (`..._2d_parity`) | `figures/fig_2d_parity.py` |
| Fig 5 defect topologies (`defect_propagation_networks_parity`) | `figures/fig_defect_topologies.py` |
| Table 1 (affine ECAs) / Table 2 (structure factors) | `data/make_tables.py` |
| 88-rule gradient table + 4 corrections | `verify_vichniac.py` |
| Claims C1–C7 | `verification/test_c1..c7_*.py` |

See [docs/provenance.md](docs/provenance.md) for exact commands, expected
results, observed status and honest caveats.

## The numerical hazard, and the fix

`numpy.linalg.matrix_power` overflows silently for moderate exponents, which is
the artefact class the paper is about. All exact powers go through
`src/lyapunov/gf2.py`. The precise behaviour (int64 magnitude overflow;
float-route parity loss; the two's-complement parity coincidence) is documented
in that module and demonstrated in `verification/test_c7_numerical_artefact.py`.

## Determinism

All RNG seeds are fixed and documented (`data/make_graphs.py`, the figure
scripts). Graphs, tables and figure numerics are byte-identical across runs.

## Licence & citation

MIT (see `LICENSE`). Please cite the article and this software; see
`CITATION.cff`.
