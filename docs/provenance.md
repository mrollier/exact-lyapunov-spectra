# Provenance: figure / claim → script → command → expected → status

Every manuscript figure, table and claim maps to a script and an exact command.
Figure and table numbers are those of the revised manuscript of 14 September
2026 (6 figures, 4 tables). "Observed" reflects a clean end-to-end run: a fresh
Python 3.11 environment with only `requirements.txt` installed,
`python reproduce.py all`, on the development laptop (12 cores) unless the
workstation is named.

Run everything with:

```bash
pip install -r requirements.txt && pip install -e .
python reproduce.py all          # figures + tables + full checks
python reproduce.py quick        # tables + checks only (CI subset)
```

Figures are written to `output/`; tables to `data/tables/`; graphs to
`data/graphs/`. The generators live in `scripts/`; `verify_vichniac.py` at the
root is a launcher for `scripts/verify_vichniac.py`, kept because App. A of
the manuscript prints `python verify_vichniac.py`.

## Figures

| Fig | Manuscript stem | Script | Command | Expected | Status |
|-----|-----------------|--------|---------|----------|--------|
| 1 | `persistent_defect_eca_diff` | `figures/fig_defect_cones.py` | `python figures/fig_defect_cones.py` | 2×4 defect-cone panels, class IV (54,110,147,124) / class III (30,101,90,150), N=51, T=100 | ✅ reproduces (RNG seed differs from original; layout & qualitative patterns match) |
| 2 | `singular_values_and_lyapunov_spectra_of_constant_J_ECAs_NO_CLASSES` | `figures/fig_eca_spectra.py` | `python figures/fig_eca_spectra.py` | 4 rows by gradient weight; σ_k vs k/N (left), log-frequency of Λ_k (right); N=3001 | ✅ reproduces (matches PDF) |
| 3 | `convergence_rule150` | `figures/make_convergence_figure.py` | `python figures/make_convergence_figure.py --rule 150 --N 101 --T 200 --burn 100 --zoom-k 35` (or `notebooks/04_convergence_figure.ipynb`, which writes `convergence_rule150_notebook`) | (A) rule 150 spectrum by 4 methods + k≤35 inset; (B) Benettin error per k with the same 200-step budget: no burn-in vs burn-in 100 + 100-step window; prints and asserts the W = 1000 reference table (fractions to 0.02, max errors to a factor of 2), including the 6.5e-5 the text quotes for B = 1000, T = 1000 | ✅ reproduces (matches PDF); ~4 s |
| 3 (supporting study) | `convergence_benettin_rule150`, `convergence_horizon_vs_N`, `convergence_direct_multiplication` | `notebooks/03_benettin_convergence.ipynb` (reference: `notebooks/fig3_convergence_study.py`) | `python notebooks/execute.py` | Benettin error vs T with/without burn-in (k = 1, 10, 51, 101; 1/T line); fraction of spectrum within 1e-2 vs T for N = 31, 61, 101, 201; unscaled float64 direct multiplication at T = 50, 100, 200, 300 with predicted floors; the comparison with Vispoel et al.'s printed histogram ranges for rules 60, 90 and 150 that App. C refers to (cells 12–13) | ✅ 73/73 in-notebook checks pass; ~35 s |
| 3, panel A alone | `benchmark_rule150` | `figures/make_benchmark_figure.py` | `python figures/make_benchmark_figure.py --rule 150 --N 101 --T 200 --zoom-k 35` | The figure of the original submission (4 methods + inset) | ✅ reproduces; superseded by the row for Fig. 3 |
| 4 | `singular_values_and_log_spectra_2d_parity` | `figures/fig_2d_parity.py` | `python figures/fig_2d_parity.py` | 3 rows (vN, Moore, r2-vN); σ_{k,l} heatmap + Λ histogram; MLE ln5/ln9/ln13; N = 201 per side | ✅ reproduces (matches PDF) |
| 5 | `damage_vs_mle` | `figures/fig_damage_vs_mle.py` | `python figures/fig_damage_vs_mle.py` | Three stacked scatter plots (shared x axis, serif) of the mean relative damage D_norm from one flipped cell (damaged cells / cells in the maximal light cone, averaged over the final steps and the initial configurations) against the maximal exponent Λ_max (single renormalised tangent vector) on the same trajectories. Top: all 88 ECAs up to symmetry (ring N=607, T=300, final 75 steps, exponent burn-in 100, 24 configurations), rules 30, 90, 110, 150 labelled; middle: all 528 outer-totalistic von Neumann rules up to conjugation (torus L=149, T=70, final 17 steps, burn-in 20, 16 configurations); bottom: 2000 of the 131 328 outer-totalistic Moore rules up to conjugation, drawn uniformly from the classes with `default_rng([20240601, 2, 8])` (same torus, T, window and burn-in, 10 configurations). Spearman ρ per panel (0.61, 0.52, 0.82; bootstrap 95 % interval [0.80, 0.84] for the sample, 2000 resamples, `default_rng(0)`); rules with Λ_max = −∞ in a marked column; dashed lines at ln 2, ln 3; ln 4, ln 5; ln 8, ln 9 | ✅ reproduces from the committed caches (matches PDF); ~5 s (the caches take ~1 min, ~10 min and ~37 min on 10 cores) |
| 6 | `lyapunov_spectra_nonaffine_ecas` | `figures/fig_nonaffine_spectra.py` | `python figures/fig_nonaffine_spectra.py` | 2x6 densities of the Lyapunov exponents: rules 6, 26, 73, 154, 41, 122, 126, 54, 110 at Vispoel's settings (N=1000, T=500, burn-in 200, 40 samples), then the affine rules 60, 90, 150 from the closed form; 50 bins shared over [-2.7, 1.2], log frequency from 1e-5, the top decade left clear for the annotation, dashed line at the maximal exponent, and per panel the share of -inf exponents plus either the MLE (the sample mean) with its 16th/84th-percentile spread or, for the three affine rules, the exact ln 2 or ln 3 | ✅ reproduces from the committed cache; ~4 s (the cache itself is 9.2 core-hours). One annotation differs from the printed manuscript: rule 110's MLE is 0.654456 in the cache, drawn as 0.654, where the manuscript prints 0.655 |
| S1 (supplementary; in the original submission, not in the revised manuscript) | `defect_propagation_networks_parity` | `figures/fig_defect_topologies.py` | `python figures/fig_defect_topologies.py` | 2×2 Ring/Grid/WS/BA defect patterns, A^t e_j (mod 2), nodes by eigenvector centrality | ✅ reproduces (seeded graphs; ring→Sierpinski, WS/BA irregular) |

## Tables

| Tab | Content | Script | Output | Status |
|-----|---------|--------|--------|--------|
| 1 | Abbreviations | — | — | typeset in the manuscript only |
| 2 | Corrected entries of Vichniac (1990), Table 1: five ECAs (62, 110, 130, 146, 172), seven entries; plus the full gradient table of the 88 non-equivalent ECAs and the comparison with the published table | `scripts/verify_vichniac.py` (`python verify_vichniac.py`) | `data/tables/gradient_corrections_table.tex` (Tab. 2), `eca_gradient_table.csv`, `vichniac_table1_computed.csv`, `vichniac_table1_diff.md` | ✅ 88 rows; 7 mismatching entries in 5 rules: 62, 110, 130, 146, 172; Tab. 2 checked by truth table against the printed values |
| 3 | 16 affine ECAs: gradients & weights, in the manuscript's row order | `scripts/make_tables.py` | `data/tables/affine_ecas.csv` | ✅ computed from core |
| 4 | Structure factor K(k,l) + parity MLE (3 neighbourhoods) | `scripts/make_tables.py` | `data/tables/structure_factors.csv` | ✅ computed from core |

### Repository tables (not typeset in the manuscript)

| Table | Content | Script | Output | Status |
|-------|---------|--------|--------|--------|
| Non-affine spectra (Fig. 6) | Twelve ECAs (nine non-affine sampled, three affine exact): MLE, sample spread (standard deviation and 16th/84th percentiles), share of -inf exponents | `scripts/make_nonaffine_spectra.py` | `data/tables/nonaffine_spectra.csv`, `data/nonaffine/spectra.npz` | ✅ 40 samples per rule; MLE 0.413 (rule 26) to 0.915 (rule 73) |
| Direct method (Fig. 6, App. C) | What the unscaled direct method returns at the same settings | `scripts/make_nonaffine_spectra.py` | `data/tables/nonaffine_direct_multiplication.csv` | ✅ 4 rules overflow float64; the other 5 report 276-364 exponents above a floor only 10-25 of them reach |
| Damage, 1-D (Fig. 5) | 88 ECAs: Λ_max (mean, sd, number of −∞ samples), v_front, D_norm, fill (means and sds), final damage and radius, whether the cone grows. Parameters (ring N = 607, T = 300, burn-in 100, window 75, 24 samples, seed 20240601) are in the `.npz` and in `scripts/make_damage_mle.py` | `scripts/make_damage_mle.py --dim 1` | `data/tables/damage_vs_mle_1d.csv`, `data/damage/damage_mle_1d.npz` | ✅ 24 samples per rule; 8 rules at −∞ (0, 8, 32, 40, 128, 136, 160, 168); Λ_max up to 1.097 (rules 150, 105) |
| Damage, von Neumann (Fig. 5) | 528 outer-totalistic von Neumann rules: the same columns, labelled in B/S notation (torus L = 149, T = 70, burn-in 20, window 17, 16 samples) | `scripts/make_damage_mle.py --dim 2` | `data/tables/damage_vs_mle_2d.csv`, `data/damage/damage_mle_2d.npz` | ✅ 16 samples per rule; 16 rules at −∞; Λ_max up to 1.597 (the parity rule B13/S024, exact ln 5 = 1.609 less the finite-horizon transient) |
| Damage, Moore (Fig. 5) | 2000 sampled outer-totalistic Moore rules (of 131 328 classes): the same columns, B/S notation with counts 0–8 (same torus and horizon, 10 samples) | `scripts/make_damage_mle.py --dim 2 --neighbourhood moore` | `data/tables/damage_vs_mle_2d_moore.csv`, `data/damage/damage_mle_2d_moore.npz` | ✅ 10 samples per rule; 15 rules at −∞ and 8 partly annihilated; Λ_max up to 2.147 (B0246/S0135); the most damage 0.498 (B246/S01246) |

Column definitions for the damage tables: `v_front` is the damage radius over
the elapsed time (front speed), `D_norm` the damaged cells over the maximal
light cone (2t + 1 on the ring, 2t² + 2t + 1 on the von Neumann torus,
(2t + 1)² on the Moore torus), `fill` the damaged cells over the cone actually
reached, each averaged over the final window and the samples; `cone_grows` is
`v_front > 0.1`.

## The 88-rule catalogue (App. C: "spectra for all 88 non-equivalent ECAs")

The same machinery over every ECA up to reflection and conjugation, at the
same settings as Fig. 6 (N = 1000, T = 500, burn-in 200, window 300, 40
samples per rule). Nine of the 88 are affine and come from the closed form; the
other 79 are sampled. Produced once, on the workstation, and not regenerated
by `reproduce.py`; the revised manuscript's App. C points readers here.
| What | Script | Command | Output | Status |
|------|--------|---------|--------|--------|
| Spectra, exact ranks and the direct-method comparison for all 88 rules | `scripts/make_nonaffine_spectra.py` | `python scripts/make_nonaffine_spectra.py --all-88 --recompute --workers 8` | `data/nonaffine/spectra_all88.npz` (14.5 MB, committed), `data/nonaffine/all88_spectra.csv` (88 rows), `data/nonaffine/all88_direct_multiplication.csv` (79 rows) | ✅ 3160/3160 samples, 696.6 min wall clock (11 h 37 min) on 2026-09-14/15; seven non-affine rules (8, 32, 40, 128, 136, 160, 168) plus affine rule 0 annihilate the tangent space, every exponent -inf; finite MLEs run from 0 (rules 4, 15, 51, 170, 204) to ln 3 (105, 150); the share of the spectrum at -inf reaches 90.1 % (rule 104); the direct method overflows float64 on all 40 samples for 22, 41, 45, 54, 73, 106, 126 and on some for 108 |

## Claims (verification suite)

Each claim has a dedicated pytest file (plus lower-level unit tests). Run
`python -m pytest -q`; exit code is non-zero on any failure.

| Claim | Statement | Test | Status |
|-------|-----------|------|--------|
| C1 | 16 affine ECAs = exactly the constant-Jacobian rules, with the tabulated gradients (recomputed from first principles) | `verification/test_c1_affine_constant_jacobian.py` | ✅ pass |
| C2 | Closed form matches a stable numerical routine across N, T (direct-mult float64 exact at the top; Benettin converges) | `verification/test_c2_benchmark.py`, `test_benettin.py` | ✅ pass |
| C3 | MLEs: rules 150/105 → ln3, rule 90 → ln2; 2-D parity ln5/ln9/ln13; Moore = 2 ln3; rule 150 has a zero singular value iff 3 divides N, so N = 3001 and N = 101 (Figs. 2, 3) give finite spectra | `verification/test_c3_mle_values.py`, `test_spectra.py` | ✅ pass |
| C4 | Recompute all 88 gradients from definition (5); compare by truth table with Vichniac's Table 1; exactly 7 misprinted entries in 5 rows (62, 110, 130, 146, 172), cross-checked by additivity and symmetry from the paper's own rows | `verification/test_c4_vichniac_gradients.py` | ✅ pass (see note) |
| C5 | Parity spectrum = ln\|λ_k(A) + a_o\| and MLE = ln ρ(A + a_o I) for both a_o (Eq. 15); ρ(A) ≥ mean degree on a connected graph, so the MLE is positive (Sec. 4.2); supplementary: single-site amplitude ∝ eigenvector centrality (WS, BA) | `verification/test_c5_parity_centrality.py` | ✅ pass |
| C6 | Benettin sanity: Σ exponents = ln|det J| at every T | `verification/test_c6_benettin_det_sum.py` | ✅ pass |
| C7 | Numerical artefacts reproduced beside the correct result (overflow; float16 plateau) | `verification/test_c7_numerical_artefact.py` | ✅ pass (see note) |
| C9 | Convergence calibration (Fig. 3): Benettin 1/T transient, burn-in, N-dependent horizon; unscaled direct method exact at the top, resolves 31/21/15/11 exponents at T = 50/100/200/300, overflows beyond T = 323; floor formula; eigenbasis start exact at T = 1 | `verification/test_benettin_convergence.py` + the notebook's final cell; windowed estimator (W = 1000 reference table, two slow bands, same-budget burn-in) in `verification/test_benettin_windowed.py` | ✅ pass |
| C10 | Spectra of the nine non-affine rules Vispoel et al. report, recomputed with Benettin on the configuration-dependent Jacobian; the number of -inf exponents is a rank deficiency, computed exactly over GF(p), not a threshold on the QR diagonal; the unscaled direct method cannot produce these spectra at these settings (band 0.036 wide for every rule; 4 of 9 overflow) | `verification/test_c10_nonaffine_spectra.py`, `verification/test_nonaffine.py` | ✅ pass (see note) |
| C11 | Boolean damage against Λ_max for every rule (Fig. 5, App. B): the single-vector exponent equals the DFT closed form on constant Jacobians to 1e-9 and lies within 0.01 of ln 3 / ln 2 on the affine ECAs and within 0.02 of the full-QR cache on the nine rules of Fig. 6; the damage never wraps and reproduces the exact mod-2 defect patterns (rule 90: 2^popcount(t)); −∞ rules are exactly 0, 8, 32, 40, 128, 136, 160, 168 on the ring and 16 rules on the torus, never a mixture; Spearman ρ(Λ_max, D_norm) = 0.61 (1-D) and 0.52 (2-D), ρ(Λ_max, v_front) = 0.14 (1-D): at ln 2 exactly the front speed runs from 0 (rule 232) to 1 (rule 90); one sample per dimension recomputes bit for bit, and a small from-scratch run reproduces the qualitative claims | `verification/test_c11_damage_vs_mle.py`, `test_damage.py`, `test_outer_totalistic.py` | ✅ pass (see note) |
| C12 | The sampled Moore family (Fig. 5, bottom panel): the 2000 rules are exactly the seeded uniform draw of distinct minimal representatives from the 131 328 classes; the single-vector exponent of the Moore parity rules equals the DFT closed form to 1e-9 and falls short of ln 9 / ln 8 by 0.013 / 0.015 (finite horizon, bounded by 0.02); Life's sample 0 is pinned (1.341872); 15 rules have every sample at −∞ and 8 have only some (a mixture never seen on the ring or the von Neumann torus); Spearman ρ(Λ_max, D_norm) = 0.82; one sample recomputes bit for bit and a from-scratch small run exercises the pipeline | `verification/test_c12_damage_vs_mle_moore.py`, `test_outer_totalistic.py`, `test_damage.py` | ✅ pass (see note) |
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
  computation and checked by truth table against the values printed in Tab. 2
  of the revised manuscript (re-read from the PDF on 15 September 2026). The
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
  match. The supplementary figure S1's WS/BA graphs use fresh fixed seeds
  (documented in `scripts/make_graphs.py`); the ring length (225) and grid side
  (15) are choices. Fig. 5's bootstrap interval uses `default_rng(0)`.
- **App. C and the three affine rules of Vispoel et al.** The statement that
  their printed MLEs for rules 60, 90 and 150 and the missing low-Λ tail result
  from direct multiplication in floating point is demonstrated in
  `notebooks/03_benettin_convergence.ipynb` (cells 12–13: the predicted float64
  floors at T = 500, 1.063 for rule 150 and 0.657 for rules 60/90, against the
  ranges of their Fig. 18; overflow beyond T = 323) and pinned in
  `verification/test_benettin_convergence.py`; it is not in a CSV table.
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
- **C11: one tangent vector, not the full spectrum.** Λ_max in Fig. 5 is the growth rate of a single renormalised tangent vector under the configuration-dependent Boolean Jacobian (Benettin with k = 1, O(N) per step), which is the top exponent and nothing else; it is the only thing affordable on a torus with L² = 22 201 tangent dimensions and it is all the figure needs. On a constant normal Jacobian its finite-time value is known in closed form from the Fourier coefficients of the starting vector, and the code reproduces that to 1e-9. What remains is the 1/T transient of the estimator: at N = 607, T = 300, burn-in 100 the affine ECAs come out within 0.0015 of ln 3 / ln 2; at L = 149, T = 70, burn-in 20 the 2-D parity rules fall short of ln 5 / ln 4 by 0.012 and 0.012, because the subdominant Fourier modes decay only as (2π/L)² per step. The tests bound that shortfall rather than hide it. The nine rules of Fig. 6 agree with their full-QR cache within 0.011 although the lattice, horizon and window all differ.
- **C11: −∞ and the damage conventions.** A rule whose Jacobian annihilates the tangent vector exactly has Λ_max = −∞ (the norm becomes exactly zero; no threshold); such rules are drawn in a separate column and excluded from Spearman's ρ. On the ring these are exactly the eight rules 0, 8, 32, 40, 128, 136, 160, 168, and every sample of such a rule dies. The damage is measured with L ≥ 2T + 3 (primes 607 and 149), so it never wraps and the three views are independent of the lattice; the primality matters only to the dense tangent vector. The 2-D cone is the L1 ball (2t² + 2t + 1 cells); `fill` is 0 when the damage has healed, and a cone with v_front ≤ 0.1 is drawn grey because its fill is then a ratio of two small numbers. The norm is taken with numpy's pairwise sum rather than BLAS `ddot`, whose summation order depends on the thread count; that is what makes the caches recomputable bit for bit from any process.
- **C12: a sample, not a census.** The outer-totalistic Moore family ("Life-like" rules) has 2¹⁸ = 262 144 rules and, since black–white conjugation f′(c, n) = 1 − f(1 − c, 8 − n) fixes 2⁹ of them, 131 328 classes. At the parameters of the von Neumann panel every class would cost about 580 CPU-hours, so Fig. 5's bottom panel is a uniform sample of 2000 classes: the sorted list of minimal representatives is enumerated (vectorised, checked against the per-rule loop on the von Neumann family) and `default_rng([20240601, 2, 8]).choice` draws 2000 without replacement, so the sample is fixed by the seed and the tests regenerate it. Each class gets 10 initial configurations (seeds `[20240601, 2, 8, rule, sample]`) rather than 16, which keeps the run near 37 minutes on 10 cores; with 2000 rules the sampling error of Spearman's ρ is what matters, and the seeded percentile bootstrap over the rules (2000 resamples) gives [0.80, 0.84] around 0.82. The cone is the Chebyshev square (2t + 1)² and the radius the Chebyshev distance, so L ≥ 2T + 3 still keeps the damage from wrapping. Two things differ from the smaller families: the exponent orders the damage much more tightly (ρ = 0.82 against 0.61 and 0.52), and 8 sampled rules are *mixtures* — the configuration usually dies to a fixed point whose Jacobian is zero, but some initial configurations leave a small surviving pattern with a finite exponent (B46/S45: 9 of 10 samples annihilated, the tenth at exactly ln 2). Their plotted exponent is the mean over the finite samples, the convention the cache has always used; the test pins which rules and how many samples.
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
- **88-rule catalogue: machine, cost and cross-machine agreement.** Run on
  2026-09-14 16:55 to 2026-09-15 04:33 on a Xeon W-2295 (18 cores / 36
  threads, 32 GB, two of four DDR4 channels populated), CPython 3.12.3 with the
  pinned numpy 1.26.4 / scipy 1.13.1 (OpenBLAS 0.3.27), eight single-threaded
  workers: 3160 samples in 696.6 min, 13.2 s per sample effective. The worker
  count was measured with `scripts/bench_workers.py` (1 worker 13.0 QR steps/s,
  4: 42.6, 6: 49.8, 8: 49.4, 12: 44.9, 18: 38.7, 36: 33.6), an effective
  speed-up of 3.8 rather than 18: the memory bandwidth, not the cores, is the
  limit, as on the laptop. The nine Fig. 6 rules are part of the catalogue
  with the same seeds, so the two runs can be compared across LAPACK builds.
  The exact ranks agree bitwise for every sample (integer arithmetic), and the
  mean maximal exponents agree to 1e-4 or better (rule 6: 0.543882 vs 0.543986;
  rule 154: 0.479598 vs 0.479556; the other seven to six decimals). Individual
  finite-time exponents do not: the Boolean trajectory is identical but the
  frame's rounding path is not, and after 500 steps single exponents of single
  samples differ by up to 0.18 above -2 and by orders of magnitude in the
  collapsed tail. The number of exact zeros on the QR diagonal differs too
  (rule 73: 2.8 % of the pivots on this build against 9.85 % on the laptop's)
  while the exact rank, and hence the count of -inf exponents, is the same,
  which is the reason the count is not read off the diagonal. On this build
  three tests that assumed bitwise agreement with the laptop needed a
  tolerance: the affine closed form against the cache (libm's complex
  exponential differs by 1e-14), and the pivot-continuum check for rules 54
  and 110 (a pivot at 1e-31 where the laptop's LAPACK returned exactly 0, and
  one N = 80 trajectory with a 2.6-decade gap; the test now pools three).
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
