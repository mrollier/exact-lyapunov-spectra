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
  vichniac.py        recompute the 88-rule gradient table; compare with Vichniac
  vichniac_table1.py Vichniac (1990) Table 1 exactly as printed (misprints kept)
figures/             one standalone script per manuscript figure -> output/
verification/        one pytest check per claim (C1–C7) + core unit tests
data/                make_graphs.py, make_tables.py, generated tables/graphs
verify_vichniac.py   CLI: gradient table, Vichniac comparison, diff report, LaTeX table
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
| Corrected-entries table (`tab:gradient-table`, 5 ECAs) + 88-rule gradient table | `verify_vichniac.py` |
| Claims C1–C7 | `verification/test_c1..c7_*.py` |

See [docs/provenance.md](docs/provenance.md) for exact commands, expected
results, observed status and honest caveats.

## Corrections to Vichniac (1990), Table 1

Table 1 of Vichniac, G. Y. (1990), *Boolean derivatives on cellular automata*,
Physica D 45, 63-74, lists the Boolean gradient of each of the 88 minimal
representative ECAs. Seven printed entries in five rows are wrong. Notation:
`x[i-1]`, `x[i]`, `x[i+1]` are the three inputs, `~` is NOT, juxtaposition is
AND and `+` is OR.

| Rule | Table    | Entry     | Printed (wrong)               | Correct                       |
|-----:|----------|-----------|-------------------------------|-------------------------------|
|   62 | 00111110 | d/dx[i-1] | ~x[i] + x[i+1]                | x[i] + ~x[i+1]                |
|  110 | 01101110 | d/dx[i-1] | ~x[i] x[i+1]                  | x[i] x[i+1]                   |
|  110 | 01101110 | d/dx[i+1] | x[i-1] + x[i]                 | x[i-1] + ~x[i]                |
|  130 | 10000010 | d/dx[i+1] | x[i+1] x[i] + ~x[i+1] ~x[i]   | x[i-1] x[i] + ~x[i-1] ~x[i]   |
|  146 | 10010010 | d/dx[i-1] | x[i] + x[i+1]                 | ~x[i] + x[i+1]                |
|  146 | 10010010 | d/dx[i+1] | x[i-1] + x[i]                 | x[i-1] + ~x[i]                |
|  172 | 10101100 | d/dx[i-1] | x[i] x[i+1] + ~x[i] ~x[i+1]   | x[i] ~x[i+1] + ~x[i] x[i+1]   |

The middle entry `d/dx[i]` is correct in all five rows; the other 83 rows are
correct.

**How the correct entries are computed.** The partial Boolean derivative of a
rule `f` with respect to input `x_k` is defined by Vichniac's eq. (5),

    df/dx_k = f(..., x_k, ...) XOR f(..., ~x_k, ...),

so it is computed exhaustively: for each of the eight neighbourhoods
`(x[i-1], x[i], x[i+1])`, flip the input in question, read both outputs off the
rule table and XOR them. This gives the truth table of the derivative, which is
a function of the two remaining inputs only (Vichniac's property (v)). The
minimal disjunctive normal form printed in the tables is obtained from that
truth table with a Quine-McCluskey minimiser; it is for display only, and every
comparison in this repository is made by truth table, never by comparing
expression strings (Vichniac's literal order is not consistent between rows).
Reading the derivative straight off the rule table in this way is Vichniac's
Method 5, the half-string technique: `d/dx[i-1]` is the XOR of the two halves
of the 8-bit rule string, `d/dx[i]` the XOR of bits 1, 2, 5, 6 with bits 3, 4,
7, 8, and `d/dx[i+1]` the XOR of the odd- and even-position bits (positions
counted from the left). The implementation is `gradient_truth_tables` in
`src/lyapunov/rules.py`.

**How the misprints were established.**

1. *Definition.* All 3 x 88 derivatives were computed from eq. (5) as above and
   compared by truth table with the published table
   (`src/lyapunov/vichniac_table1.py`, transcribed from the Physica D scan and
   verified against it by aligned rendering and pixel-level overbar counting).
   Exactly the seven entries above differ (`verify_vichniac.py`).
2. *Additivity cross-check from the paper's own rows* (Vichniac's property
   (iii), his Method 4). If `n = n1 XOR n2` as rule numbers, each gradient
   component of `n` is the XOR of the corresponding components of `n1` and
   `n2`. Using only correctly printed rows, 62 = 60 XOR 2, 110 = 106 XOR 4,
   130 = 128 XOR 2, 146 = 128 XOR 18 and 172 = 168 XOR 4 reproduce the correct
   entries and differ from the printed rows in exactly the seven slots, so the
   corrections follow from Vichniac's table without trusting our computation.
3. *Property (v).* The printed `d/dx[i+1]` of rule 130 contains `x[i+1]`
   itself, which no partial derivative can; it is a subscript typo.

**Probable origin.** Rules 62, 110, 146 and 172 are related to correctly
printed rows by Vichniac's Method 3 (complementing variables and/or
reflecting): `f62(L,C,R) = ~f56(L,~C,~R)`, `f110(L,C,R) = ~f44(R,~C,L)`,
`f146(L,C,R) = f104(L,~C,R)` and `f172(L,C,R) = ~f78(R,~C,L)`. In each case
the printed row is the source row (reflected where applicable) with the
variable complementation omitted: entirely for 110, 146 and 172, and in the
first slot only for 62. The printed 172 entry is the XNOR of `x[i]` and
`x[i+1]` where the XOR is correct. Rule 130 has no such explanation.

These checks are encoded in `verification/test_c4_vichniac_gradients.py`.
`python verify_vichniac.py` writes `data/tables/vichniac_table1_computed.csv`
(all correct entries), `data/tables/vichniac_table1_diff.md` (the seven
differences) and `data/tables/gradient_corrections_table.tex` (the manuscript's
corrected-entries table), and ends with the line
`7 mismatching entries in 5 rules: 62, 110, 130, 146, 172`.

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
