# Provenance: figure / claim → script → command → expected → status

Every manuscript figure, table and claim maps to a script and an exact command.
"Observed" reflects a clean end-to-end run: a fresh Python 3.11 environment with
only `requirements.txt` installed, `python reproduce.py all`, on 2025 hardware.

Run everything with:

```bash
pip install -r requirements.txt && pip install -e .
python reproduce.py all          # figures + tables + full checks
python reproduce.py quick        # tables + checks only (CI subset)
```

Figures are written to `output/`; tables to `data/tables/`; graphs to
`data/graphs/`.

## Figures

| Fig | Manuscript stem | Script | Command | Expected | Status |
|-----|-----------------|--------|---------|----------|--------|
| 1 | `persistent_defect_eca_diff` | `figures/fig_defect_cones.py` | `python figures/fig_defect_cones.py` | 2×4 defect-cone panels, class IV (54,110,147,124) / class III (30,101,90,150), N=51, T=100 | ✅ reproduces (RNG seed differs from original; layout & qualitative patterns match) |
| 2 | `singular_values_and_lyapunov_spectra_of_constant_J_ECAs_NO_CLASSES` | `figures/fig_eca_spectra.py` | `python figures/fig_eca_spectra.py` | 4 rows by gradient weight; σ_k vs k/N (left), log-frequency of Λ_k (right); N=3001 | ✅ reproduces (matches PDF) |
| 3 | `benchmark_rule150` | `figures/make_benchmark_figure.py` | `python figures/make_benchmark_figure.py --rule 150 --N 101 --T 200 --zoom-k 35` | Rule 150 spectrum by 4 methods + k≤35 inset | ✅ reproduces (matches PDF); superseded in revision by the row below |
| 3 (revised figure) | `convergence_rule150` | `figures/make_convergence_figure.py` | `python figures/make_convergence_figure.py --rule 150 --N 101 --T 200 --burn 100 --zoom-k 35` (or `notebooks/04_convergence_figure.ipynb`) | (A) the original 4-method benchmark + inset; (B) Benettin error per k with the same 200-step budget: no burn-in vs burn-in 100 + 100-step window; prints and asserts the W = 1000 reference table (fractions to 0.02, max errors to a factor of 2) | ✅ reproduces; ~4 s |
| 3 (supporting study) | `convergence_benettin_rule150`, `convergence_horizon_vs_N`, `convergence_direct_multiplication` | `notebooks/03_benettin_convergence.ipynb` (reference: `notebooks/fig3_convergence_study.py`) | `python notebooks/execute.py` | Benettin error vs T with/without burn-in (k = 1, 10, 51, 101; 1/T line); fraction of spectrum within 1e-2 vs T for N = 31, 61, 101, 201; unscaled float64 direct multiplication at T = 50, 100, 200, 300 with predicted floors | ✅ 73/73 in-notebook checks pass; ~35 s |
| 6 | `lyapunov_spectra_nonaffine_ecas` | `figures/fig_nonaffine_spectra.py` | `python figures/fig_nonaffine_spectra.py` | 2x6 densities of the Lyapunov exponents: rules 6, 26, 73, 154, 41, 122, 126, 54, 110 at Vispoel's settings (N=1000, T=500, burn-in 200, 40 samples), then the affine rules 60, 90, 150 from the closed form; 50 bins shared over [-2.7, 1.2], log frequency from 1e-5, the top decade left clear for the annotation, dashed line at the maximal exponent, and per panel the share of -inf exponents plus either the MLE (the sample mean) with its 16th/84th-percentile spread or, for the three affine rules, the exact ln 2 or ln 3 | ✅ reproduces from the committed cache; ~4 s (the cache itself is 9.2 core-hours) |
| 4 | `singular_values_and_log_spectra_2d_parity` | `figures/fig_2d_parity.py` | `python figures/fig_2d_parity.py` | 3 rows (vN, Moore, r2-vN); σ_{k,l} heatmap + Λ histogram; MLE ln5/ln9/ln13 | ✅ reproduces (matches PDF) |
| 5 | `defect_propagation_networks_parity` | `figures/fig_defect_topologies.py` | `python figures/fig_defect_topologies.py` | 2×2 Ring/Grid/WS/BA defect patterns, A^t e_j (mod 2), nodes by eigenvector centrality | ✅ reproduces (fresh seeds; ring→Sierpinski, WS/BA irregular) |

## Tables

| Tab | Content | Script | Output | Status |
|-----|---------|--------|--------|--------|
| T1 | 16 affine ECAs: gradients & weights | `data/make_tables.py` | `data/tables/affine_ecas.csv` | ✅ computed from core |
| T2 | Structure factor K(k,l) + parity MLE (3 neighbourhoods) | `data/make_tables.py` | `data/tables/structure_factors.csv` | ✅ computed from core |
| T4 | Twelve ECAs (nine non-affine sampled, three affine exact): MLE, sample spread (standard deviation and 16th/84th percentiles), share of -inf exponents | `data/make_nonaffine_spectra.py` | `data/tables/nonaffine_spectra.csv`, `data/nonaffine/spectra.npz` | ✅ 40 samples per rule; MLE 0.413 (rule 26) to 0.915 (rule 73) |
| T5 | What the unscaled direct method returns at the same settings | `data/make_nonaffine_spectra.py` | `data/tables/nonaffine_direct_multiplication.csv` | ✅ 4 rules overflow float64; the other 5 report 276-364 exponents above a floor only 10-25 of them reach |
| T3 | Gradient DNF for all 88 non-equivalent ECAs; comparison with Vichniac (1990) Table 1; corrected-entries table (5 ECAs, 7 entries) | `verify_vichniac.py` | `data/tables/eca_gradient_table.csv`, `vichniac_table1_computed.csv`, `vichniac_table1_diff.md`, `gradient_corrections_table.tex` | ✅ 88 rows; 7 mismatching entries in 5 rules: 62, 110, 130, 146, 172 |

## Claims (verification suite)

Each claim has a dedicated pytest file (plus lower-level unit tests). Run
`python -m pytest -q`; exit code is non-zero on any failure.

| Claim | Statement | Test | Status |
|-------|-----------|------|--------|
| C1 | 16 affine ECAs = exactly the constant-Jacobian rules, with the tabulated gradients (recomputed from first principles) | `verification/test_c1_affine_constant_jacobian.py` | ✅ pass |
| C2 | Closed form matches a stable numerical routine across N, T (direct-mult float64 exact at the top; Benettin converges) | `verification/test_c2_benchmark.py`, `test_benettin.py` | ✅ pass |
| C3 | MLEs: rules 150/105 → ln3, rule 90 → ln2; 2-D parity ln5/ln9/ln13; Moore = 2 ln3 | `verification/test_c3_mle_values.py` | ✅ pass |
| C4 | Recompute all 88 gradients from definition (5); compare by truth table with Vichniac's Table 1; exactly 7 misprinted entries in 5 rows (62, 110, 130, 146, 172), cross-checked by additivity and symmetry from the paper's own rows | `verification/test_c4_vichniac_gradients.py` | ✅ pass (see note) |
| C5 | Parity MLE = ln ρ(A); single-site amplitude ∝ eigenvector centrality (WS, BA) | `verification/test_c5_parity_centrality.py` | ✅ pass |
| C6 | Benettin sanity: Σ exponents = ln|det J| at every T | `verification/test_c6_benettin_det_sum.py` | ✅ pass |
| C7 | Numerical artefacts reproduced beside the correct result (overflow; float16 plateau) | `verification/test_c7_numerical_artefact.py` | ✅ pass (see note) |
| C9 | Convergence calibration (revised Fig. 3): Benettin 1/T transient, burn-in, N-dependent horizon; unscaled direct method exact at the top, resolves 31/21/15/11 exponents at T = 50/100/200/300, overflows beyond T = 323; floor formula; eigenbasis start exact at T = 1 | `verification/test_benettin_convergence.py` + the notebook's final cell; windowed estimator (W = 1000 reference table, two slow bands, same-budget burn-in) in `verification/test_benettin_windowed.py` | ✅ pass |
| C10 | Spectra of the nine non-affine rules Vispoel et al. report, recomputed with Benettin on the configuration-dependent Jacobian; the number of -inf exponents is a rank deficiency, computed exactly over GF(p), not a threshold on the QR diagonal; the unscaled direct method cannot produce these spectra at these settings (band 0.036 wide for every rule; 4 of 9 overflow) | `verification/test_c10_nonaffine_spectra.py`, `verification/test_nonaffine.py` | ✅ pass (see note) |
| C8 | (nilpotency) | — | ⛔ out of scope: not in the submission (only an open-question comment atop the .tex) |

### Notes / honest caveats

- **C4 scope.** Vichniac's original Table 1 is transcribed in
  `src/lyapunov/vichniac_table1.py` (verified against the Physica D scan on
  12 September 2026; misprints deliberately kept). All 3 x 88 entries are
  compared by truth table with the derivative computed from definition (5);
  seven entries in five rows differ. The corrections are cross-checked using
  only correctly printed rows of the same table (additivity, Method 4) and
  their probable origin is reproduced (symmetry, Method 3, with the variable
  complementation omitted; rule 130 is a subscript typo violating property
  (v)). The manuscript's corrected-entries table is generated from the
  computation and checked by truth table against the submitted values. The
  88-row recomputed table is internally exact (every minimised DNF matches its
  truth table). Two equally minimal covers exist for phi of rule 62; the
  generator prints `s[i+1]~s[i]` where the manuscript prints `s[i+1]~s[i-1]`
  as the third product. Both are correct and the gradient is unaffected.
- **C7 int64 subtlety.** The paper flags int64 `matrix_power` as a hazard for
  `A^t (mod 2)`. Precisely: a *pure int64* power wraps modulo 2^64 and 2 | 2^64,
  so `matrix_power(int64) % 2` accidentally keeps the correct parity; only the
  magnitudes are wrong. The parity is destroyed when the power is formed in
  **floating point** (float64 loses the low bit above 2^53) and then reduced —
  the realistic bug. `gf2.gf2_matrix_power` is correct in every case and is what
  the defect-pattern code uses. Both regimes are demonstrated in the C7 test.
- **Figure seeds.** Figure 1 uses a fixed RNG seed (20240601) for its random
  initial configuration; the original figure's seed was not published, so the
  exact defect pattern differs while the rules, layout and qualitative structure
  match. Figure 5's WS/BA graphs use fresh fixed seeds (documented in
  `data/make_graphs.py`); the ring length (200) and grid side (15) are choices,
  as the manuscript does not state them.
- **C10: counting the -infinite exponents.** For eight of the nine rules the
  Boolean Jacobian is singular at almost every step, so part of the tangent
  space is annihilated and those exponents are exactly -infinity. How many is a
  rank question, and it is answered exactly: the window product is accumulated
  modulo a prime (three products of residues stay well inside float64, so the
  reduction is exact) and eliminated over GF(p), cross-checked against a second
  prime and, on windows short enough for floating point to be trusted, against
  `numpy.linalg.matrix_rank`. It is deliberately *not* read off the QR diagonal.
  An unpivoted QR is not rank-revealing, and for rules 122, 126, 54 and 110 the
  near-zero pivots run continuously from 1e-6 down to 1e-18 with no internal gap
  wider than 1.4 decades, so no threshold could be defended; where the collapse
  instead comes from a vanishing gradient row (rules 6, 73, 41) the pivots do
  fall to 1e-33 and below and a gap does exist. The two counts agree for seven
  of the nine rules; for rules 26 and 122 the floating-point iteration returns a
  large finite exponent for directions that are in fact annihilated. The figure
  therefore censors by rank, not by value, which is unambiguous: the censored
  exponents always lie below every kept one.
- **C10: what the direct method does here.** The precision floor is
  `ln(sigma_max) + ln(eps)/(2T)`, so the resolvable band is `|ln(eps)|/(2T)`
  wide *whatever the rule is*: 0.036 at T = 500. Rules 73, 41, 126 and 54
  overflow float64 outright, since the largest eigenvalue of `Y Y^T` is of order
  `exp(2 T MLE)` and reaches 1e398, 1e374, 1e309 and 1e322 against a float64
  ceiling of 1.8e308; all 40 samples of each overflow, though rule 126 only just.
  For the remaining
  five the method returns 276 to 364 exponents above a floor that only 10 to 25
  of the true exponents reach, and returns a further 442 to 469 of the 1000 as
  `nan` because their eigenvalues come out negative. A histogram of that is a
  spike just below the maximal exponent, which is the signature of the floor and
  not a property of the rule.
- **C10: the three affine rules, and what the figure cuts off.** Rules 60, 90
  and 150 are drawn from the closed form of `lyapunov.spectra`, not from a
  trajectory: one exact vector each, no sampling and no burn-in, with maximal
  exponents exactly ln 2, ln 2 and ln 3. Their zero singular values are counted
  the same exact way as everything else, and the count agrees with the algebra
  (the symbol vanishes at a root of unity: k = N/2 for rule 60 since N is even,
  k = N/4 and 3N/4 for rule 90 since 4 divides N, and nowhere for rule 150 since
  3 does not divide 1000). That censoring matters here for a second reason: the
  closed form returns a rounding-level number, not zero, where a singular value
  is algebraically zero, so its logarithm would otherwise appear as a spurious
  -36. Their spectra also have far longer left tails than the non-affine ones,
  because ln(sigma) falls steeply as a singular value passes near zero: they
  reach -5.07, -4.38 and -5.62 against -2.80 for the worst non-affine rule. The
  figure's shared range is cut at -2.7 so every panel stays legible, which
  leaves out 2.0 %, 2.0 % and 2.4 % of those three spectra and exactly one
  exponent (0.003 %) of rule 6; nothing else is affected. `main` prints the
  counts on every run.
- **C10: budget and finite size.** Burn-in 200 with a window of 300 is
  converged: against a five times longer budget the maximal exponent moves by at
  most 0.011 (rule 126; typically under 0.005) and the share of -infinite
  exponents is unchanged for six of the nine rules and within 1.3 points for the
  rest. Sample-to-sample spread of the maximal exponent over the 40 initial
  configurations runs from 0.0012 (rule 41) to 0.025 (rule 73). Finite size is
  the larger effect: between N = 150 and N = 1000 rule 73's maximal exponent
  moves by 0.106, and the ordering of neighbouring rules is not stable (110 and
  122 differ by 0.005; 41 and 73 exchange places). The split into a slow group
  (26, 154, 6; below 0.6) and a fast group (the other six; above 0.6) is stable,
  and that is what the reduced-size check in the test suite asserts.
- **C10: cost and parallelism.** One sample is about 92 s on one core (65 s of
  QR, 16 s for the exact rank, 11 s for the direct method), so the run is 9.2
  core-hours. Ten workers do not give a tenfold speed-up, because ten
  simultaneous 1000 x 1000 QR factorisations saturate the memory bandwidth; the
  observed wall clock was about three and a half hours. Each sample is seeded
  independently (`default_rng([20240601, rule, sample])`) and each worker runs
  single-threaded BLAS, so the result does not depend on the worker count. The
  cache is committed; `reproduce.py all` only redraws from it, in about three
  seconds, and never triggers the computation.
- **Determinism.** Graphs, tables and figure numerics are byte-identical across
  runs (no unse­eded randomness in the core).
- **Convergence notebook: BLAS threads and quoted digits.** The notebook and
  its reference script pin the BLAS thread count to one before importing numpy.
  With several threads the QR rounding pathway differs and the fraction of the
  N = 201 spectrum within 1e-2 moves at the second decimal (0.94 vs 0.95 at
  T = 800: thirteen exponents sit within 2e-3 of the threshold). The trend is
  unaffected. Two numbers quoted in the working notes for this figure were
  off in the last digit and are corrected by the computation: the predicted
  float64 floor at T = 500 for rules 60/90 is 0.657 (not 0.656), and the
  "median interior singular-value ratio 0.9994" is (1 + 0.9987)/2 because rule
  150 on an odd ring has 50 exactly degenerate pairs; the slow convergence of
  the median exponent is a late swap between eigenpairs of opposite sign and
  nearly equal modulus (gap ratio 0.9689), not an alignment rate of 0.9994.
