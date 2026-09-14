"""
Reference implementation for the Fig. 3 convergence study.

INDEPENDENT CROSS-CHECK, DELIBERATELY NOT IMPORTING src/lyapunov.
It re-derives the rule-150 Jacobian, the closed form and the two numerical
routes from scratch so that notebooks/03_benettin_convergence.ipynb can assert
the core package against a second implementation. Do not "de-duplicate" it into
the core: its value is that it shares no code with what it checks.

Notes
* ``numpy.linalg.matrix_power`` is used here in float64 on purpose: the float64
  rounding of the unscaled product is the artefact under study, not something to
  avoid (the CLAUDE.md hazard concerns exact integer powers; see
  ``lyapunov.gf2`` for those).
* BLAS threading changes the rounding pathway of QR and therefore the last
  digits of the slowest-converging exponents (visible in the N = 201 fraction
  at the 1e-2 level). The thread count is pinned to one before numpy is
  imported so that two runs give identical numbers.

Original docstring follows.

Compares three routes to the Lyapunov spectrum of an affine ECA with a
constant, circulant (hence normal) Jacobian J:

  exact     Lambda_k = ln|lambda_k(J)| = ln sigma_k(J), closed form (Eq. 11 for rule 150)
  Benettin  QR re-orthonormalisation every step, from Q0 = I, with and without burn-in
  direct    Y^T = J^T, exponents from eigenvalues of Y Y^T  (the method of Vispoel et al. 2024)

Everything is float64 unless stated.  Reference numbers are printed at the end
so a notebook can assert against them.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

# ---------------------------------------------------------------- ECA machinery
def table(rule):
    """Lookup table t[4l + 2c + r] = phi(l, c, r) for a Wolfram rule number."""
    return np.array([(rule >> k) & 1 for k in range(8)], dtype=np.int8)

def jacobian(t, s):
    """Boolean Jacobian on a periodic ring, as a dense float 0/1 matrix.
    For an affine rule the result is independent of s."""
    N = len(s); l, r = np.roll(s, 1), np.roll(s, -1)
    base = t[4*l + 2*s + r]
    dm = base ^ t[4*(1-l) + 2*s + r]        # sensitivity to the left neighbour
    d0 = base ^ t[4*l + 2*(1-s) + r]        # to the cell itself
    dp = base ^ t[4*l + 2*s + (1-r)]        # to the right neighbour
    J = np.zeros((N, N)); i = np.arange(N)
    J[i, (i-1) % N] = dm; J[i, i] = d0; J[i, (i+1) % N] = dp
    return J

def exact_rule150(N):
    """Eq. (11): Lambda_k = ln|1 + 2 cos(2 pi k / N)|, sorted descending."""
    k = np.arange(1, N+1)
    return np.sort(np.log(np.abs(1 + 2*np.cos(2*np.pi*k/N))))[::-1]

# ---------------------------------------------------------------- the three routes
def benettin(J, T, burn, checkpoints, Q0=None):
    """Running-average Lyapunov estimates at the given checkpoints.

    burn  : QR steps taken before accumulation starts (frame alignment only)
    Q0    : starting orthonormal frame (identity if None)
    """
    N = len(J); Q = np.eye(N) if Q0 is None else Q0.copy()
    for _ in range(burn):
        Q, _ = np.linalg.qr(J @ Q)
    acc = np.zeros(N); out = {}
    for tt in range(1, T+1):
        Q, R = np.linalg.qr(J @ Q)
        acc += np.log(np.abs(np.diag(R)))
        if tt in checkpoints:
            out[tt] = acc / tt
    return out

def direct_multiplication(J, T):
    """Vispoel et al. (2024), Eqs. (8)-(10), (35): Y^T = J^T, then eigenvalues of Y Y^T.
    Float64.  Overflows once |lambda_max|^(2T) exceeds ~1.8e308."""
    Y = np.linalg.matrix_power(J, T)                 # float64: no int64 wraparound
    ev = np.sort(np.linalg.eigvalsh(Y @ Y.T))[::-1]
    out = np.full(len(ev), np.nan)              # negative eigenvalues: rounding noise, no logarithm
    out[ev > 0] = np.log(ev[ev > 0]) / (2*T)
    return out

def precision_floor(lam_max, T, eps=2.2e-16):
    """Every exponent below this reads as the floor in double precision."""
    return np.log(lam_max) + np.log(eps) / (2*T)

# ---------------------------------------------------------------- reference numbers
if __name__ == "__main__":
    N = 101; J = jacobian(table(150), np.zeros(N, dtype=np.int8)); ex = exact_rule150(N)
    sv = np.exp(ex)
    print(f"exact: top {ex[0]:.5f} (ln3 = {np.log(3):.5f}), bottom {ex[-1]:.5f}, "
          f"negative and finite: {(ex < 0).sum()}/{N}")
    print(f"consecutive ratios: |l2/l1| = {sv[1]/sv[0]:.4f}, median interior = {np.median(sv[1:]/sv[:-1]):.4f}")

    Ts = [50, 100, 200, 500, 1000, 2000]
    for burn, lab in [(0, "Q0 = I, no burn-in"), (200, "200-step burn-in")]:
        res = benettin(J, 2000, burn, set(Ts))
        print(f"\nBenettin, {lab}:   T   top      k=10     k=51     bottom")
        for T in Ts:
            e = np.abs(res[T] - ex)
            print(f"   {T:5d}  {e[0]:.2e} {e[9]:.2e} {e[50]:.2e} {e[100]:.2e}")

    print("\ndirect multiplication, float64:")
    for T in (50, 100, 200, 300):
        d = direct_multiplication(J, T); fl = precision_floor(3, T)
        print(f"   T={T:3d}: floor {fl:.3f}, exponents resolved above floor+0.02: {int(np.nansum(d > fl+0.02))}/{N}, "
              f"negative eigenvalues (no result): {int(np.isnan(d).sum())}")
    print(f"   overflow of Y Y^T for rule 150 beyond T = {int(np.log(np.finfo(float).max)/(2*np.log(3)))}")
    print(f"   predicted floors at Vispoel's T=500: rule 150 {precision_floor(3,500):.3f}, rules 60/90 {precision_floor(2,500):.3f}")

    # eigenbasis start: QR is a no-op, exact at every step
    w, X = np.linalg.eigh(J); o = np.argsort(-np.abs(w)); w, X = w[o], X[:, o]
    res = benettin(J, 5, 0, {1, 5}, Q0=X)
    print(f"\nfrom the eigenbasis: max |Lambda - exact| at T=1: {np.abs(res[1]-ex).max():.1e}, at T=5: {np.abs(res[5]-ex).max():.1e}")

# ---------------------------------------------------------------- reference data file
def reference_numbers(path="fig3_reference.json"):
    """Recompute every number the notebook asserts against and write them to JSON.
    Runs in well under a minute; this is the single source of truth."""
    import json
    N = 101; J = jacobian(table(150), np.zeros(N, dtype=np.int8)); ex = exact_rule150(N)
    sv = np.exp(ex); Ts = [25, 50, 100, 200, 400, 800, 1600, 3200]; idx = {"k1": 0, "k10": 9, "k51": 50, "k101": 100}
    nob = benettin(J, 3200, 0, set(Ts)); bur = benettin(J, 3200, 200, set(Ts))
    # N-dependence.  NOTE: a single index such as the median exponent is NOT a
    # robust measure (its local spectral gap varies non-monotonically with N).
    # Use the fraction of the whole spectrum converged to within 1e-2 instead.
    ndep = {}
    for n in (31, 61, 101, 201):
        Jn = jacobian(table(150), np.zeros(n, dtype=np.int8)); exn = exact_rule150(n)
        r = benettin(Jn, 3200, 200, set(Ts))
        ndep[str(n)] = [float((np.abs(r[T] - exn) < 1e-2).mean()) for T in Ts]
    dm = {}
    for T in (50, 100, 200, 300):
        d = direct_multiplication(J, T); fl = precision_floor(3, T)
        dm[str(T)] = {"floor": float(fl), "resolved_above_floor": int(np.nansum(d > fl + 0.02)),
                      "n_negative_eigenvalues": int(np.isnan(d).sum()),
                      "positive_noise_max_dist_from_floor": float(np.nanmax(np.abs(d[d <= fl + 0.02] - fl)))}
    w, X = np.linalg.eigh(J); o = np.argsort(-np.abs(w)); X = X[:, o]
    eig = benettin(J, 5, 0, {1, 5}, Q0=X)
    out = {
        "exact": {"mle": float(ex[0]), "bottom": float(ex[-1]), "n_negative_finite": int((ex < 0).sum()),
                  "ratio_l2_l1": float(sv[1]/sv[0]), "median_interior_ratio": float(np.median(sv[1:]/sv[:-1]))},
        "T": Ts,
        "benettin_noburn_abs_err": {k: [float(abs(nob[T][i] - ex[i])) for T in Ts] for k, i in idx.items()},
        "benettin_burn200_abs_err": {k: [float(abs(bur[T][i] - ex[i])) for T in Ts] for k, i in idx.items()},
        "fraction_within_1e-2_vs_N_burn200": ndep,
        "direct_multiplication_float64": dm,
        "overflow_T_rule150": int(np.log(np.finfo(float).max) / (2*np.log(3))),
        "predicted_floor_T500": {"rule150": float(precision_floor(3, 500)), "rules60_90": float(precision_floor(2, 500))},
        "eigenbasis_start_max_err": {"T1": float(np.abs(eig[1]-ex).max()), "T5": float(np.abs(eig[5]-ex).max())},
    }
    json.dump(out, open(path, "w"), indent=1)
    return out

if __name__ == "__main__":
    import time; t0 = time.time()
    ref = reference_numbers()
    print(f"\nwrote fig3_reference.json in {time.time()-t0:.0f}s")
