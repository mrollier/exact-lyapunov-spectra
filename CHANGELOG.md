# Changelog

## 1.3.0 (2026-09-13)

### Lyapunov spectra of nine non-affine ECAs (new Fig. 6, claim C10)

Rules 6, 26, 73, 154, 41, 122, 126, 54 and 110 at the settings of Vispoel et al.
(2024), Section 6: N = 1000 on a ring, T = 500, 40 random initial
configurations, with a burn-in of 200 and a window of 300.

- New module `src/lyapunov/nonaffine.py`, the only core addition. Existing
  modules are untouched apart from the package index and the version. It
  provides the Boolean Jacobian at a given configuration (`eca_jacobian_at`,
  equal to `eca_jacobian` on the affine rules), a banded propagation
  (`apply_bands`, O(N^2) instead of O(N^3), which is what makes N = 1000
  affordable), Benettin along a trajectory
  (`benettin_log_stretch_trajectory`), the unscaled direct method for a
  time-varying Jacobian (`direct_multiplication_trajectory`), and the exact
  rank of the tangent map (`tangent_rank`).
- The number of exponents equal to -infinity is a rank deficiency and is
  computed exactly over GF(p), not by thresholding the QR diagonal. An
  unpivoted QR is not rank-revealing: for rules 122, 126, 54 and 110 the
  near-zero pivots run continuously from 1e-6 to 1e-18 with no internal gap
  wider than 1.4 decades, and for rules 26 and 122 the floating-point iteration
  returns a large finite exponent for directions that are in fact annihilated.
  Rule 154 is the only one of the nine with no annihilated directions, because
  its derivative with respect to the right neighbour is the constant 1.
- Maximal exponents 0.413 (rule 26) to 0.915 (rule 73); the share of the
  spectrum at -infinity ranges from 0 % (rule 154) to 41.9 % (rule 73). The
  figure and `data/tables/nonaffine_spectra.csv` report the spread of the 40
  sample maxima as the 16th and 84th percentiles about their mean, which is
  asymmetric where it matters: rule 73 runs from -0.010 to +0.047.
- The unscaled direct method cannot produce these spectra at these settings.
  The resolvable band is |ln(eps)|/(2T) = 0.036 wide for every rule; rules 73,
  41, 126 and 54 overflow float64 on all 40 samples; the other five place 276
  to 364 exponents above a floor that only 10 to 25 of the true exponents
  reach, and return 442 to 469 of the 1000 as nan.
- Rules 60, 90 and 150 are drawn beside them from the closed form (no
  trajectory, no sampling): maximal exponents exactly ln 2, ln 2 and ln 3, with
  0.1 %, 0.2 % and 0 % of their spectra at -infinity. Their zero singular values
  are counted by the same exact rank, which is cross-checked against the
  algebraic count for all 16 affine rules over nine ring sizes and both primes.
- New `data/make_nonaffine_spectra.py` (parallel, seeded per sample, cached in
  `data/nonaffine/spectra.npz`), `figures/fig_nonaffine_spectra.py`, the tables
  `data/tables/nonaffine_spectra.csv` and
  `data/tables/nonaffine_direct_multiplication.csv`, and the checks
  `verification/test_nonaffine.py` (112) and
  `verification/test_c10_nonaffine_spectra.py` (39). The suite is now 374 tests
  and takes about 110 s.
- `data/make_nonaffine_spectra.py --all-88` extends the same machinery to every
  ECA up to reflection and conjugation: 79 by trajectory and 9 affine ones from
  the closed form, into their own cache and tables, leaving the manuscript's
  numbers untouched. A run of that size takes hours, so it checkpoints every
  100 samples and `--resume` continues from the checkpoint; because each sample
  is seeded from `(seed, rule, sample)` alone, a resumed run is bitwise
  identical to an uninterrupted one, which two tests pin. `data/bench_workers.py`
  measures how many single-threaded workers the machine can actually feed: the
  bottleneck is memory bandwidth, not cores, and on the development laptop the
  effective speed-up saturated at 2.4 on eight workers of twelve.

## 1.2.0 (2026-09-13)

### Convergence study replacing Fig. 3 (referee 2, paragraph 3)

- New `notebooks/03_benettin_convergence.ipynb`: the exact affine spectrum of
  rule 150 (N = 101) calibrates Benettin's algorithm (1/T frame-alignment
  transient, burn-in, N-dependent horizon measured as the fraction of the
  spectrum within 1e-2) and the unscaled direct-multiplication method of
  Vispoel et al. (2024) (precision floor ln 3 + ln(eps)/(2T) rising with T;
  31/21/15/11 exponents resolved at T = 50/100/200/300; float64 overflow beyond
  T = 323; predicted floors at T = 500 of 1.063 for rule 150 and 0.657 for
  rules 60/90). Eigenbasis start makes QR a no-op (C3 triangle and rule 150);
  a constant non-normal (lower bidiagonal) Jacobian shows why Eq. (7) uses
  ln|lambda_k|. Three figures to `output/convergence_*.pdf`; 73 in-notebook
  checks against `notebooks/fig3_convergence_study.py`, an independent
  reimplementation, whose reference JSON is regenerated into `output/`.
- `src/lyapunov/benettin.py`: added `benettin_running_average` (burn-in,
  starting frame, checkpoints), `direct_multiplication_unscaled` (Vispoel's
  Eqs. 8-10 and 35 verbatim, `nan` for eigenvalues of Y Y^T that rounding has
  made negative, raises `OverflowError` once sigma_max^(2T) exceeds float64)
  and `precision_floor`. Existing functions unchanged.
- `verification/test_benettin_convergence.py`: 14 unit tests for the above.
- New `figures/make_convergence_figure.py` (stem `convergence_rule150`): the
  original benchmark panel over a panel of the per-exponent Benettin error at a
  fixed 200-step budget, no burn-in vs burn-in 100 + 100-step window. Prints
  and asserts the W = 1000 reference table (B = 0, 100, 300, 1000, 3000 and
  the running average at T = 4000). Core: `benettin_log_stretch` (one run
  stores every per-step log stretch), `windowed_spectrum`,
  `cumulative_spectrum`; tests in `verification/test_benettin_windowed.py`.
- New `notebooks/04_convergence_figure.ipynb` builds that figure interactively:
  every layout number is a key of `make_convergence_figure.STYLE`, the stored QR
  run is reused when the burn-in or horizon changes, and the final cell asserts
  the reference table. `notebooks/execute.py` now runs both notebooks and takes
  notebook names as arguments.
- `notebooks/execute.py` runs the notebook headlessly; `reproduce.py all` calls
  it (skipped with a message if the new optional `notebook` extra is absent).
- Finding while verifying the working notes: the "median interior ratio 0.9994"
  is the median over 100 consecutive singular-value ratios of which 50 are
  exactly 1 (degenerate pairs k, N - k), i.e. (1 + 0.9987)/2; the median
  exponent converges slowly because its frame vector sits in the neighbouring
  eigenpair (opposite sign, modulus 3 % smaller) until step ~900 and then
  swaps, after which the running average carries the transient as 1/T.

## 1.1.0 (2026-09-12)

### Corrections to Vichniac (1990), Table 1: five rules, seven entries

- Rule 172 added to the list of misprinted rows (62, 110, 130, 146, 172). Its
  printed `d/dx[i-1]` is the XNOR `x[i] x[i+1] + ~x[i] ~x[i+1]`; the correct
  entry is the XOR `x[i] ~x[i+1] + ~x[i] x[i+1]`.
- Audit finding: the previous release did not store Vichniac's published table
  at all. It hard-coded only the four corrected entries taken from the
  manuscript (`PAPER_CORRECTIONS`) and checked those against the computation,
  so rule 172 was never compared and the omission could not be detected. The
  recomputed 88-row table already contained the correct XOR for 172.
- New `src/lyapunov/vichniac_table1.py`: the published table transcribed
  exactly as printed (verified against the Physica D scan on 12 September
  2026), misprints deliberately kept; the single source of truth for "what
  Vichniac printed". `PAPER_CORRECTIONS`, `check_corrections` and
  `paper_correction_truth_tables` are removed; the corrected values are now
  computed, never transcribed.
- `src/lyapunov/vichniac.py` compares all 3 x 88 entries by truth table and
  reports the mismatches; it also derives gradients from related rows by
  symmetry (Method 3) and additivity (Method 4) for the cross-checks, and
  generates the manuscript's corrected-entries table (`tab:gradient-table`,
  five ECAs).
- `verify_vichniac.py` keeps `--output` and the column layout of
  `eca_gradient_table.csv`, and additionally writes
  `vichniac_table1_computed.csv`, `vichniac_table1_diff.md` and
  `gradient_corrections_table.tex`. Its summary ends with
  `7 mismatching entries in 5 rules: 62, 110, 130, 146, 172` and the exit code
  is non-zero if the mismatch set differs from the documented seven.
- `verification/test_c4_vichniac_gradients.py` rewritten: exact mismatch set,
  additivity and symmetry cross-checks from the paper's own rows, property (v)
  on printed and computed entries (only 130, `d/dx[i+1]` fails), unit-weight
  and affine sanity checks, and a truth-table check of the manuscript table.
- README, `docs/provenance.md` and docstrings updated from "four corrections"
  to five rules and seven entries.

## 1.0.0

Initial release of the reproducibility package.
