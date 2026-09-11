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
| 3 | `benchmark_rule150` | `figures/make_benchmark_figure.py` | `python figures/make_benchmark_figure.py --rule 150 --N 101 --T 200 --zoom-k 35` | Rule 150 spectrum by 4 methods + k≤35 inset | ✅ reproduces (matches PDF) |
| 4 | `singular_values_and_log_spectra_2d_parity` | `figures/fig_2d_parity.py` | `python figures/fig_2d_parity.py` | 3 rows (vN, Moore, r2-vN); σ_{k,l} heatmap + Λ histogram; MLE ln5/ln9/ln13 | ✅ reproduces (matches PDF) |
| 5 | `defect_propagation_networks_parity` | `figures/fig_defect_topologies.py` | `python figures/fig_defect_topologies.py` | 2×2 Ring/Grid/WS/BA defect patterns, A^t e_j (mod 2), nodes by eigenvector centrality | ✅ reproduces (fresh seeds; ring→Sierpinski, WS/BA irregular) |

## Tables

| Tab | Content | Script | Output | Status |
|-----|---------|--------|--------|--------|
| T1 | 16 affine ECAs: gradients & weights | `data/make_tables.py` | `data/tables/affine_ecas.csv` | ✅ computed from core |
| T2 | Structure factor K(k,l) + parity MLE (3 neighbourhoods) | `data/make_tables.py` | `data/tables/structure_factors.csv` | ✅ computed from core |
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
- **Determinism.** Graphs, tables and figure numerics are byte-identical across
  runs (no unse­eded randomness in the core).
