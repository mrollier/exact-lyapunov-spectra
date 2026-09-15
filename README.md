# Exact Lyapunov spectra of affine cellular automata and the parity rule on networks

Reproducibility package for the article *"Exact Lyapunov spectra of affine
cellular automata and the parity rule on networks"* by Michiel Rollier and
Jan M. Baetens (BionamiX, Ghent University), submitted to *Chaos, Solitons &
Fractals* (revised version of 14 September 2026). This is the repository the
article's data-availability statement refers to:
<https://github.com/mrollier/exact-lyapunov-spectra>.

Every figure, table and quantitative claim in the paper is reproducible from this
repository with a single command, and the central claims are independently
verified by a test suite. Figure and table numbers below are those of the
revised manuscript (6 figures, 4 tables, appendices A to C).

## What the paper shows

Affine Boolean rules — those whose update is an `XOR` of a subset of the inputs
plus a constant — have a **configuration-independent Boolean Jacobian**. Their
Lyapunov spectrum is therefore *exact*: the logarithms of the singular values of
one constant matrix, with no simulation and no limit. On a periodic lattice that
matrix is (multilevel) circulant, so the spectrum is the discrete Fourier
transform of the gradient stencil; for the parity rule on any graph it is the
adjacency matrix (plus the identity for the self-inclusive rule), so the
spectrum is the logarithm of the absolute graph spectrum and the maximal
exponent the logarithm of its spectral radius. The exact spectra serve as
benchmarks: they calibrate Benettin's algorithm and expose the precision floor
of direct multiplication (Fig. 3, App. C), correct seven misprints in the
gradient table of Vichniac (1990) (Tab. 2, App. A), and are set against the
Boolean damage of a single flipped cell for all 88 ECAs, all 528
outer-totalistic von Neumann rules and 2000 sampled outer-totalistic Moore
rules (Fig. 5, App. B).

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
python figures/make_convergence_figure.py --rule 150 --N 101 --T 200 --burn 100 --zoom-k 35
python verify_vichniac.py --output data/tables/eca_gradient_table.csv
python -m pytest -q
python notebooks/execute.py        # both Fig. 3 notebooks (needs '.[notebook]'; rewrites them in place)
python notebooks/execute.py 04_convergence_figure    # just the figure notebook
python figures/fig_nonaffine_spectra.py              # Fig. 6, from the committed cache
python data/make_nonaffine_spectra.py --recompute    # regenerate that cache (9.2 core-hours)
python data/bench_workers.py                         # how many workers this machine can feed
python data/make_nonaffine_spectra.py --all-88 --recompute --workers 8   # all 88 ECAs (App. C; ~12 h, see docs/provenance.md)
python figures/fig_damage_vs_mle.py                  # Fig. 5, from the committed caches
python data/make_damage_mle.py --dim 1 --recompute   # 88 ECAs (~1 min)
python data/make_damage_mle.py --dim 2 --recompute   # 528 outer-totalistic vN rules (~10 min on 10 cores)
python data/make_damage_mle.py --dim 2 --neighbourhood moore --recompute   # 2000 sampled Moore classes (~37 min on 10 cores)
python figures/fig_defect_topologies.py              # supplementary figure S1 (not in the manuscript)
```

## Repository layout

```
src/lyapunov/        verified maths core (imported by figures and tests)
  gf2.py             int64-safe GF(2) / integer matrix powers (the central fix)
  rules.py           ECA rule tables, Boolean gradients, affine detection, 88 classes
  jacobian.py        circulant / adjacency Jacobians; pure-numpy ECA evolution
  spectra.py         closed-form DFT singular values; neighbourhood structure factor
  benettin.py        Benettin QR + direct-multiplication reference methods
  parity.py          parity rule on graphs: spectrum, MLE, A^t e_j (mod 2)
  nonaffine.py       configuration-dependent Jacobian, banded Benettin, exact rank (Fig. 6)
  outer_totalistic.py 2-D outer-totalistic rules, von Neumann and Moore (Fig. 5)
  damage.py          Boolean damage from one flipped cell; single-vector MLE (Fig. 5)
  quine_mccluskey.py Boolean minimiser for the gradient table
  vichniac.py        recompute the 88-rule gradient table; compare with Vichniac
  vichniac_table1.py Vichniac (1990) Table 1 exactly as printed (misprints kept)
figures/             one standalone script per manuscript figure -> output/
                     (+ fig_defect_topologies.py, supplementary figure S1)
notebooks/           03_benettin_convergence.ipynb: the Fig. 3 convergence study
                     (+ fig3_convergence_study.py, an independent reference
                     implementation it asserts against); 04_convergence_figure.ipynb:
                     Fig. 3 built interactively; execute.py runs both headlessly
verification/        pytest checks of the claims C1-C12 (C8 is out of scope; C9 lives in
                     test_benettin_convergence.py and test_benettin_windowed.py) + unit tests
data/                make_tables.py (Tab. 3, 4), make_nonaffine_spectra.py (Fig. 6, App. C),
                     make_damage_mle.py (Fig. 5), make_graphs.py (S1), bench_workers.py;
                     the generated tables/ and the committed caches nonaffine/, damage/, graphs/
verify_vichniac.py   CLI: gradient table, Vichniac comparison, diff report, Tab. 2 as LaTeX
reproduce.py         single entry point (all | quick)
docs/provenance.md   figure/claim -> script -> command -> expected -> status
```

## Figure and claim map

| Manuscript object | Script / test |
|---|---|
| Fig. 1 defect cones (`persistent_defect_eca_diff`) | `figures/fig_defect_cones.py` |
| Fig. 2 affine ECA spectra (`..._NO_CLASSES`) | `figures/fig_eca_spectra.py` |
| Fig. 3 benchmark + Benettin error per k (`convergence_rule150`) | `figures/make_convergence_figure.py`, or `notebooks/04_convergence_figure.ipynb` to retune it (writes `convergence_rule150_notebook`) |
| Fig. 3 supporting convergence study (`convergence_benettin_rule150`, `convergence_horizon_vs_N`, `convergence_direct_multiplication`) | `notebooks/03_benettin_convergence.ipynb` |
| Fig. 3, panel A alone (`benchmark_rule150`, the figure of the original submission) | `figures/make_benchmark_figure.py` (superseded) |
| Fig. 4 2-D parity (`..._2d_parity`) | `figures/fig_2d_parity.py` |
| Fig. 5 damage vs maximal exponent, 88 ECAs, 528 outer-totalistic vN rules, 2000 sampled Moore rules (`damage_vs_mle`) | `figures/fig_damage_vs_mle.py`, computed by `data/make_damage_mle.py` |
| Fig. 6 corrected spectra of nine non-affine ECAs (`lyapunov_spectra_nonaffine_ecas`) | `figures/fig_nonaffine_spectra.py`, computed by `data/make_nonaffine_spectra.py` |
| Tab. 2 corrected entries of Vichniac (1990), Table 1 (`tab:gradient-table`, 5 ECAs) + the 88-rule gradient table | `verify_vichniac.py` |
| Tab. 3 (affine ECAs) / Tab. 4 (structure factors) | `data/make_tables.py` |
| App. C: spectra of all 88 ECAs | `data/nonaffine/all88_spectra.csv`, `spectra_all88.npz`; `data/make_nonaffine_spectra.py --all-88` |
| Claims C1–C7 | `verification/test_c1..c7_*.py` |
| Claim C9 (convergence calibration, Fig. 3) | `verification/test_benettin_convergence.py`, `verification/test_benettin_windowed.py` |
| Claim C10 (non-affine spectra, Fig. 6) | `verification/test_c10_nonaffine_spectra.py`, `verification/test_nonaffine.py` |
| Claim C11 (damage vs maximal exponent, Fig. 5) | `verification/test_c11_damage_vs_mle.py`, `verification/test_damage.py`, `verification/test_outer_totalistic.py` |
| Claim C12 (the sampled Moore family, Fig. 5) | `verification/test_c12_damage_vs_mle_moore.py` |
| Supplementary figure S1, defect propagation on ring / grid / WS / BA graphs (`defect_propagation_networks_parity`; in the original submission, not in the revised manuscript) | `figures/fig_defect_topologies.py`, graphs from `data/make_graphs.py`, checks in `verification/test_c5_parity_centrality.py` |

See [docs/provenance.md](docs/provenance.md) for exact commands, expected
results, observed status and honest caveats.

## The convergence notebook (behind Fig. 3)

`notebooks/03_benettin_convergence.ipynb` uses the exact affine spectrum of rule
150 (N = 101) to calibrate the two numerical routes to a Lyapunov spectrum, in
answer to referee 2, paragraph 3. Benettin's algorithm from the identity has a
1/T frame-alignment transient that a burn-in removes at the ends of the
spectrum, and an N-dependent horizon in the interior (the fraction of the
spectrum recovered to 1e-2 at T = 200 falls from 1.00 at N = 31 to 0.81 at
N = 201). Direct multiplication as stated by Vispoel et al. (2024), unscaled in
float64, resolves exponents only above a floor ln 3 + ln(eps)/(2T) that rises
with T (31, 21, 15, 11 exponents at T = 50, 100, 200, 300) and overflows beyond
T = 323. Starting Benettin in the eigenbasis of the (normal) Jacobian makes
every QR step a no-op; for a constant non-normal Jacobian the Schur basis plays
that role and ln(sigma_k) is not the spectrum (the manuscript's Eq. 7).

Every number in the notebook's text is printed by the cell beneath it, and the
final cell asserts 73 of them against `notebooks/fig3_convergence_study.py`, an
independent reimplementation that shares no code with `src/lyapunov`. Runtime is
about half a minute with the BLAS thread count pinned to one (the first cell
does this; multithreaded QR on 100 x 100 matrices is several times slower and
changes the rounding pathway of the slowest-converging exponents). Execute it
with `python notebooks/execute.py` after `pip install -e '.[notebook]'`, or open
it in Jupyter. Its three figures go to `output/convergence_benettin_rule150.pdf`,
`convergence_horizon_vs_N.pdf` and `convergence_direct_multiplication.pdf`.
Executing a notebook rewrites it in place with fresh outputs, which is how the
committed outputs are kept current; expect the two `.ipynb` files to show up in
`git status` afterwards.

## The nine non-affine rules (Fig. 6, App. C)

Vispoel et al. (2024), Section 6, report spectra for rules 6, 26, 73, 154, 41,
122, 126, 54 and 110 at N = 1000, T = 500 with 40 random initial configurations.
None of the nine is affine, so the Boolean Jacobian changes at every step and the
closed form does not apply; `src/lyapunov/nonaffine.py` supplies the
configuration-dependent Jacobian, a banded propagation that makes N = 1000
affordable, and Benettin along the trajectory. The spectra below use the same
budget, split as a burn-in of 200 and a window of 300. The spread is the 16th
and 84th percentile of the 40 sample maxima, as offsets from their mean; it is
what Figure 6 annotates, and it is asymmetric because a configuration that keeps
more of the tangent space alive also stretches faster.

| rule | maximal exponent | spread over 40 samples | share of the spectrum at -infinity |
|---|---|---|---|
| 6 | 0.5440 | +0.019 / -0.019 | 23.3 % |
| 26 | 0.4131 | +0.008 / -0.008 | 16.3 % |
| 73 | 0.9154 | +0.047 / -0.010 | 41.9 % |
| 154 | 0.4796 | +0.017 / -0.012 | 0.0 % |
| 41 | 0.8624 | +0.001 / -0.001 | 16.6 % |
| 122 | 0.6497 | +0.008 / -0.012 | 16.4 % |
| 126 | 0.7109 | +0.018 / -0.016 | 22.8 % |
| 54 | 0.7407 | +0.002 / -0.002 | 18.1 % |
| 110 | 0.6545 | +0.003 / -0.004 | 15.6 % |

and the three affine rules drawn beside them, from the closed form rather than a
trajectory:

| rule | maximal exponent | spread | share of the spectrum at -infinity |
|---|---|---|---|
| 60 | 0.6931, exactly ln 2 | none, exact | 0.1 %: one exponent, at k = N/2 |
| 90 | 0.6931, exactly ln 2 | none, exact | 0.2 %: two, at k = N/4 and 3N/4 |
| 150 | 1.0986, exactly ln 3 | none, exact | 0.0 %: 3 does not divide 1000 |

Two things are worth stating plainly.

**The -infinite exponents are counted exactly, not thresholded.** Eight of the
nine rules have a singular Jacobian at almost every step, so part of the tangent
space is annihilated. The number of directions this happens to is a rank, and it
is computed over a finite field (`tangent_rank`), because an unpivoted QR is not
rank-revealing: for rules 122, 126, 54 and 110 the near-zero pivots run
continuously from 1e-6 to 1e-18 with no gap at which to cut, and for rules 26 and
122 the floating-point iteration reports a large finite exponent for directions
that are in fact annihilated. Rule 154 is the exception with no annihilated
directions at all, because its derivative with respect to the right neighbour is
the constant 1, so no row of its Jacobian can vanish.

**The unscaled direct method cannot produce these spectra at these settings.**
Its precision floor lies `|ln(eps)| / (2T) = 0.036` below the maximal exponent,
and that width does not depend on the rule. Rules 73, 41, 126 and 54 overflow
float64 before the horizon is reached, on all 40 samples. For the other five the
method places 276 to 364 exponents above a floor that only 10 to 25 of the true
exponents reach, and returns another 442 to 469 of the 1000 as `nan`. Plotted,
that is a spike just below the maximal exponent: the signature of the floor, not
a property of the rule. `data/tables/nonaffine_direct_multiplication.csv` has the
numbers per rule.

**All 88 rules.** App. C also refers to the spectra of every ECA up to
reflection and conjugation at the same settings (N = 1000, T = 500, burn-in
200, 40 samples), 79 of them sampled and the 9 affine ones from the closed form.
They are committed as `data/nonaffine/all88_spectra.csv` (one row per rule),
`all88_direct_multiplication.csv` and the per-sample cache `spectra_all88.npz`
(14.5 MB), produced once with `python data/make_nonaffine_spectra.py --all-88
--recompute --workers 8` (11 h 37 min on an 18-core workstation; machine and
cross-build comparison in `docs/provenance.md`). `reproduce.py` does not
regenerate them.

## Damage against the maximal exponent, every rule (Fig. 5, App. B)

Fig. 5 asks how the tangent-space exponent relates to what a single flipped
cell actually does in configuration space, for all 88 ECAs up to reflection and
conjugation (ring N = 607, T = 300, 24 random initial configurations), all
528 outer-totalistic von Neumann rules up to conjugation (torus L = 149, T = 70,
16 configurations), and 2000 of the 131 328 outer-totalistic Moore rules up to
conjugation, drawn uniformly from the classes with a fixed seed (same torus and
horizon, 10 configurations; the full family would cost ~580 CPU-hours). The
sides are primes not smaller than 2T + 3, so the damage never wraps. The figure plots the normalised damage `D_norm` (damaged cells over
the maximal light cone, 2t + 1 on the ring, 2t² + 2t + 1 on the von Neumann
torus and (2t + 1)² on the Moore torus,
averaged over the final steps); the tables also record how far the damage
reaches (`v_front`, radius per step) and how dense it is (`fill`, damaged cells
over the cone actually reached). The exponent is the growth rate of a single renormalised
tangent vector under the configuration-dependent Boolean Jacobian
(`src/lyapunov/damage.py`, `src/lyapunov/outer_totalistic.py`), the top exponent
and nothing else, which is what makes the 2-D catalogue with 22 201 tangent
dimensions affordable; on the affine rules it reproduces the closed form to
rounding, and its 1/T transient is bounded by the tests (0.0015 in 1-D, 0.012 in
2-D). Rules whose Jacobian annihilates the vector exactly have exponent -inf:
on the ring exactly rules 0, 8, 32, 40, 128, 136, 160 and 168.

The exponent orders the total damage only loosely on the ring and the von
Neumann torus (Spearman 0.61 and 0.52) but much more tightly on the Moore torus
(0.82, bootstrap 95 % interval [0.80, 0.84] over the sampled rules, 2000
resamples seeded with 0 in `figures/fig_damage_vs_mle.py`), and the front speed
hardly at all in 1-D (0.14): at ln 2 exactly, the
front speed runs from 0 (rule 232) to 1 (rule 90). The 2-D parity rules have the
largest exponents and next to no damage, their defect pattern being
Sierpinski-like, while Life's B3/S23 on the von Neumann neighbourhood has
exponent 1.10 and a front that barely moves. On the Moore torus 8 sampled rules
are mixtures: the configuration usually dies to a fixed point with a zero
Jacobian, but some initial configurations leave a small surviving pattern with a
finite exponent; their plotted exponent is the mean over the finite samples.
Per-rule numbers are in `data/tables/damage_vs_mle_1d.csv`, `damage_vs_mle_2d.csv`
and `damage_vs_mle_2d_moore.csv`.

## Corrections to Vichniac (1990), Table 1 (Tab. 2 and App. A of the manuscript)

Table 1 of Vichniac, G. Y. (1990), *Boolean derivatives on cellular automata*,
Physica D 45, 63-74, lists the Boolean gradient of each of the 88 minimal
representative ECAs. Seven printed entries in five rows are wrong. The
manuscript's Tab. 2 gives the corrected gradients; it is generated here as
`data/tables/gradient_corrections_table.tex` and checked by truth table in
`verification/test_c4_vichniac_gradients.py`. Notation:
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
differences) and `data/tables/gradient_corrections_table.tex` (Tab. 2 of the
manuscript), and ends with the line
`7 mismatching entries in 5 rules: 62, 110, 130, 146, 172`.

## The numerical hazard, and the fix

`numpy.linalg.matrix_power` overflows silently for moderate exponents, which is
the artefact class the paper is about. All exact powers go through
`src/lyapunov/gf2.py`. The precise behaviour (int64 magnitude overflow;
float-route parity loss; the two's-complement parity coincidence) is documented
in that module and demonstrated in `verification/test_c7_numerical_artefact.py`.

## Determinism

All RNG seeds are fixed and documented (`data/make_graphs.py`, the data and
figure scripts; the repository's seed is 20240601). Graphs, tables and figure
numerics are byte-identical across runs on one machine. Across machines the
exact integer results (defect patterns, ranks, the count of -inf exponents)
are identical, while floating-point results depend on the BLAS/LAPACK build
and thread count in the last digits: the tests state a tolerance wherever that
matters and say why, and the two bit-for-bit checks against the damage caches
(`test_c11`, `test_c12`) avoid BLAS in the one reduction that would otherwise
depend on the thread count. The notebooks pin the BLAS thread count to one.

## Licence & citation

MIT (see `LICENSE`). Please cite the article and this software (version 1.6.0,
tag `v1.6.0`, accompanies the revised manuscript); see `CITATION.cff`.
