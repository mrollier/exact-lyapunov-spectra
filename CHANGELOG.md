# Changelog

## 1.6.1 (2026-09-15)

### Layout

- The generator scripts move out of `data/` and the repository root into
  `scripts/`: `verify_vichniac.py`, `make_tables.py`, `make_graphs.py`,
  `make_damage_mle.py`, `make_nonaffine_spectra.py` and `bench_workers.py`.
  `data/` now holds data only (the tables and the committed caches) and the
  scripts write into it as before; the commands in the README and in
  `docs/provenance.md` use the new paths. A launcher `verify_vichniac.py`
  stays at the root because App. A of the manuscript prints
  `python verify_vichniac.py`; it hands over to `scripts/verify_vichniac.py`
  with the same arguments.
- The README opens with a table that takes each pointer in the manuscript
  (the data-availability statement, Secs. 3 and 4.2, Figs. 2 to 6 and
  appendices A, B and C) to the script, notebook, table or test it lands on.
- No number, figure or test changes.

## 1.6.0 (2026-09-15)

### Alignment with the revised manuscript (resubmitted 14 September 2026)

The repository was audited against the resubmitted manuscript (6 figures, 4
tables, appendices A to C). Every number the manuscript quotes from the
repository was re-checked and matches, with one exception on the manuscript
side: Fig. 6 prints rule 110's MLE as 0.655 where the committed cache gives
0.654456, drawn as 0.654 (noted in `docs/provenance.md`). The changes below
are documentation, hygiene and a few behaviour-preserving code fixes.

- Figure and table numbers now follow the revised manuscript everywhere: the
  damage figure is Fig. 5 (was 7), the non-affine spectra stay Fig. 6, the
  corrected Vichniac entries are Tab. 2, the affine ECAs Tab. 3 and the
  structure factors Tab. 4. The defect-propagation-on-networks figure of the
  original submission is no longer in the manuscript and is kept as
  supplementary figure S1; the eigenvector-centrality material
  (`lyapunov.parity`, `test_c5`) is labelled supplementary and the README no
  longer presents it as a claim of the paper. The 88-rule catalogue, which
  App. C now cites, is listed in the figure/claim map. (Entries below this
  one keep the numbering of their time: "Fig. 7" there is today's Fig. 5
  and "Fig. 5" the supplementary figure S1.)
- `data/tables/affine_ecas.csv` follows the row order of Tab. 3 (0, 15, 85,
  51, 60, 102, 90, 150, each with its complement).
- Three statements of the manuscript that the suite did not cover are now
  tested: the parity MLE is `ln rho(A + a_o I)` for the self-inclusive rule
  too, the spectral radius of a connected graph is at least its mean degree
  (Sec. 4.2), and rule 150 has a zero singular value iff 3 divides N, so
  N = 3001 and N = 101 give finite spectra (Sec. 3.2).
- `lyapunov.quine_mccluskey` finds a minimum cover exactly for up to four
  variables (the greedy cover was one product too long for four of the 256
  three-variable functions; in the gradient table only the `phi` of rule 126
  changes, from four products to three; no gradient entry is affected). A
  test checks minimality exhaustively against brute force.
- Behaviour-preserving fixes in the core: `ot_from_bs` validates with a
  `ValueError` instead of an `assert`; `direct_multiplication_unscaled` no
  longer relies on short-circuit order to avoid an unbound variable; the dead
  `spectra.parity_2d_lyapunov_spectrum` and `outer_totalistic.N_RULES` are
  removed; docstrings corrected (`benettin` module index, the GF(p) bound in
  `nonaffine`, the transcription limitation in `vichniac_table1`).
- Figure scripts: the unused `--no-tex` flag becomes a working `--tex`;
  `fig_damage_vs_mle.py` documents the bootstrap seed behind the quoted
  interval; `make_benchmark_figure.py` draws float64 direct
  multiplication as red crosses, as the manuscript does, and is marked
  superseded; `make_graphs.load_graph` says so when it regenerates a cache.
- Notebooks: the diagnostic cell of `04_convergence_figure.ipynb` no longer
  prints the kernel's absolute path; the notebook saves its figure under its
  own stem (`convergence_rule150_notebook`) instead of overwriting the
  manuscript file; both notebooks re-executed. `execute.py` documents that it
  rewrites the notebooks in place.
- Tests: `verification/conftest.py` puts `data/` on the path once (no
  `sys.path` edits in test modules); the unreachable skips in `test_c6` are
  assertions; unused imports removed.
- Metadata: version 1.6.0 in `pyproject.toml`, `CITATION.cff` (was 1.2.0)
  and the package; `CITATION.cff` gains `type`, `date-released`, `url` and
  the submission status; LICENSE years 2025-2026; CI runs on Python 3.10 to
  3.12 and fails if `reproduce.py quick` changes a committed table.

### The 88-rule catalogue, run

- `data/make_nonaffine_spectra.py --all-88 --recompute --workers 8` on a Xeon
  W-2295 workstation: 79 sampled rules x 40 samples plus the 9 affine rules
  from the closed form, 3160 samples in 11 h 37 min, eight single-threaded
  workers (measured optimum; the effective speed-up over one core is 3.8 on
  18 cores, the memory bandwidth being the limit). Outputs
  `data/nonaffine/all88_spectra.csv` and
  `data/nonaffine/all88_direct_multiplication.csv` and the 14.5 MB per-sample
  cache `spectra_all88.npz`, all committed so that the repository is
  self-contained. Machine, timings and the cross-build comparison are in
  `docs/provenance.md`.
- Seven non-affine rules (8, 32, 40, 128, 136, 160, 168) annihilate the tangent
  space at every configuration reached, as rule 0 does: every exponent is
  -inf, the rank is zero and the MLE is -inf. Five rules have an MLE of
  exactly 0 (4 and the affine 15, 51, 170, 204). The finite MLEs run up to
  ln 3 (105, 150); the share of the spectrum at -inf reaches 90.1 % (rule
  104). The direct method overflows float64 on every sample for rules 22, 41,
  45, 54, 73, 106 and 126 and on some for 108.
- The nine Fig. 6 rules recomputed on a different LAPACK build (OpenBLAS
  0.3.27) agree with the committed cache in every exact rank and in the mean
  MLE to 1e-4; single finite-time exponents of single samples do not, by up
  to 0.18. The count of exact zeros on the QR diagonal differs between builds
  while the exact rank does not.
- Three tests assumed bitwise agreement with the development laptop and now
  carry a tolerance: `test_the_affine_rules_are_the_exact_closed_form`
  (`allclose` at 1e-12, -inf positions still exact) and
  `test_no_threshold_separates_the_collapsed_pivots` (pivots below 1e-25 are
  hard zeros; the continuum is pooled over three initial configurations).
- `WORKSTATION_RUN.md`, the working note for the run, is deleted now that the
  run is done.
## 1.5.0 (2026-09-14)

### Fig. 7 gains the outer-totalistic Moore rules, sampled (claim C12)

The "Life-like" family — outer-totalistic rules on the Moore neighbourhood,
2^18 = 262 144 rules in 131 328 classes under conjugation — is too large to
enumerate at the parameters of the von Neumann panel (about 580 CPU-hours),
so a third panel shows a uniform sample of 2000 classes at the same torus
(L = 149), horizon (T = 70), window and burn-in, with 10 initial
configurations each (~37 min on 10 cores). The sample is the seeded draw
`default_rng([20240601, 2, 8]).choice` from the sorted representatives and
is regenerated by the tests.

- `src/lyapunov/outer_totalistic.py` takes a `neighbourhood` argument
  (`VON_NEUMANN` by default, so nothing existing changes; `MOORE` adds the
  nine-band Jacobian, B/S labels with counts 0–8, the vectorised
  conjugation and the two Moore parity rules with exponents ln 9 and ln 8).
  `src/lyapunov/damage.py` gains the Chebyshev cone (2t + 1)^2 and radius
  (`moore=True`). The runner has `--neighbourhood moore --sample-rules N`
  and writes `data/damage/damage_mle_2d_moore.npz` and
  `data/tables/damage_vs_mle_2d_moore.csv`.
- The exponent orders the damage much more tightly on the Moore torus
  (Spearman 0.82, bootstrap 95 % interval [0.80, 0.84] over the sampled
  rules) than on the ring (0.61) or the von Neumann torus (0.52). The largest
  exponent in the sample is 2.147 (B0246/S0135) against ln 9 = 2.197 for the
  parity rule, and the most damage 0.498 (B246/S01246).
- New in this family: besides 15 rules with every sample at -inf, 8 rules
  are mixtures whose configuration usually dies to a fixed point with a zero
  Jacobian but sometimes leaves a small surviving pattern (B46/S45: nine
  samples annihilated, the tenth at exactly ln 2). Their exponent is the
  mean over the finite samples, as the cache has always defined it, and the
  tests pin which rules and how many samples.
- Checks: `verification/test_c12_damage_vs_mle_moore.py` (the sample, the
  cache, Spearman, the mixtures, the parity closed form to 1e-9 with the
  finite-horizon shortfall 0.013 / 0.015 bounded by 0.02, Life's pinned
  exponent, bit-for-bit recomputes, a from-scratch small run) plus Moore
  cases in `test_outer_totalistic.py` (200 seeded random rules against
  brute-force flips, Life's blinker, class count 131 328 with 512 fixed
  points) and `test_damage.py` (the Moore parity damage 1, 9, 9, 25, 9 is
  the tensor square of rule 150's). The suite is now 512 tests, ~130 s.

## 1.4.0 (2026-09-14)

### Boolean damage against the maximal exponent, every rule (new Fig. 7, claim C11)

For all 88 ECAs up to reflection and conjugation and all 528 outer-totalistic
von Neumann rules up to conjugation, the damage caused by flipping one cell
is put against the maximal Lyapunov exponent of the Boolean Jacobian on the
same trajectories: 1-D on a ring of N = 607 for T = 300 steps with 24 random
initial configurations, 2-D on a torus of L = 149 for T = 70 steps with 16.
Both sides are primes not smaller than 2T + 3, so the damage never wraps.

- Two new core modules, existing ones untouched apart from the package index
  and the version. `src/lyapunov/outer_totalistic.py` encodes the 1024
  outer-totalistic rules on the von Neumann neighbourhood (bit 5c + n is
  f(c, n); B/S notation; the black-white conjugation with its 32 fixed points,
  hence 528 classes), steps them on the torus, and evaluates their five-band
  configuration-dependent Boolean Jacobian, checked against brute-force flips
  for every rule and against the torus adjacency for the two parity rules.
  `src/lyapunov/damage.py` measures the Hamming distance and the damage radius
  from one flipped cell, normalises them by the light cone in three ways
  (`v_front` = radius / t, `D_norm` = damage / maximal cone, `fill` = damage /
  observed cone), and estimates the maximal exponent from a single
  renormalised tangent vector (Benettin with k = 1, O(N) per step), which is
  what makes the 2-D catalogue with L^2 = 22 201 tangent dimensions
  affordable. On an affine rule that estimate is known in closed form from
  the Fourier coefficients of the starting vector and is reproduced to 1e-9;
  the affine ECAs come out within 0.0015 of ln 3 / ln 2 and the nine rules of
  Fig. 6 within 0.011 of their full-QR cache. The 2-D parity rules fall short
  of ln 5 / ln 4 by 0.012: the 1/T transient of the estimator at T = 70,
  bounded by the tests rather than hidden by a tolerance.
- A rule whose Jacobian annihilates the tangent vector exactly has exponent
  -inf (the norm becomes exactly zero; no threshold is involved). On the ring
  these are exactly rules 0, 8, 32, 40, 128, 136, 160 and 168; on the torus 16
  rules. Every sample of such a rule dies or none does. The figure draws them
  in a separate column and Spearman's rho excludes them.
- The figure plots `D_norm` against the exponent, one panel per dimension;
  the front speed and the fill are in the tables only. The exponent orders the
  total damage only loosely (Spearman 0.61 in 1-D, 0.52 in 2-D) and the front
  speed hardly at all in 1-D (0.14): at ln 2 exactly, the front speed runs from
  0 (rule 232, the majority rule) to 1 (rule 90). The 2-D parity rules have the largest exponents and
  next to no damage, their defect pattern being Sierpinski-like (rule 90
  damages 2^popcount(t) cells and the 2-D parity rule 1, 5, 5, 17, 5, ...),
  while Life's B3/S23 on the von Neumann neighbourhood has exponent 1.10 with
  a front that barely moves. The most damaging rules are 122 (1-D, D_norm
  0.43) and B13/S02 (2-D, 0.49).
- New `data/make_damage_mle.py --dim 1|2` (parallel, seeded per sample and
  per tangent vector, caches of per-sample summaries in `data/damage/`, tables
  `data/tables/damage_vs_mle_{1,2}d.csv`), `figures/fig_damage_vs_mle.py`, and
  the checks `verification/test_outer_totalistic.py`, `test_damage.py` and
  `test_c11_damage_vs_mle.py`. The tangent-vector norm is taken with numpy's
  pairwise sum rather than BLAS `ddot`, whose summation order depends on the
  thread count, so one sample per dimension is recomputed bit for bit inside
  the test suite whatever the machine's BLAS threading. The suite is now 478
  tests and takes about 140 s.

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
