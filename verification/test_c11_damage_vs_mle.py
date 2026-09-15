"""Claim C11: Boolean damage against the maximal Lyapunov exponent, for every
ECA up to symmetry and every outer-totalistic von Neumann rule up to
conjugation (Fig. 5 of the manuscript; ``data/tables/damage_vs_mle_1d.csv``
and ``damage_vs_mle_2d.csv``).

* The maximal exponent is the growth rate of one renormalised tangent vector
  under the configuration-dependent Boolean Jacobian (Benettin with k = 1):
  on the affine rules it is within 0.01 of the exact ``ln 3`` / ``ln 2`` / 0,
  and on the nine rules of Fig. 6 within 0.02 of the full-QR cache of C10,
  although the lattice, horizon and window all differ.
* The damage of a single flipped cell never wraps (``L >= 2T + 3``), so the
  three light-cone views are lattice-independent; the affine rules give the
  exact mod-2 defect patterns (rule 90: ``2**popcount(t)`` damaged cells).
* Rules that annihilate the tangent vector have exponent ``-inf``; on the ring
  these are exactly 0, 8, 32, 40, 128, 136, 160, 168, none of them a
  mixture: every sample of a rule dies or none does.
* Across rules the exponent orders the total damage only loosely (Spearman
  0.61 in 1-D, 0.52 in 2-D), and the front speed hardly at all in 1-D (0.14):
  at ``ln 2`` exactly the front speed spans the whole interval from 0 (rule
  232) to 1 (rule 90). The 2-D parity rules have the largest exponents and
  next to no damage (their defect pattern is Sierpinski-like), and Life's
  B3/S23 on the von Neumann neighbourhood has exponent 1.10 with a damage
  radius that barely moves.

The reference values are the committed caches, and the manuscript quotes
them. The determinism of the runner is checked by recomputing one sample per
dimension and comparing bit for bit, and the qualitative claims are reproduced
from scratch at a small size, so a stale cache cannot make this file pass.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from lyapunov.outer_totalistic import (
    PARITY_VN_EXCLUSIVE, PARITY_VN_INCLUSIVE, nonequivalent_outer_totalistic)
from lyapunov.rules import affine_ecas, nonequivalent_ecas
from lyapunov.spectra import VON_NEUMANN_2D, eca_mle

from make_damage_mle import (  # data/ is on sys.path via conftest.py
    CONE_GROWS, DEFAULTS, SEED, cache_path, compute, load, run_sample, summary_rows,
    tangent_rng)

# The nine rules of Fig. 6 and their cached full-QR maximal exponents (C10).
FIG6_MLE = {6: 0.543986, 26: 0.413080, 73: 0.915411, 154: 0.479556, 41: 0.862367,
            122: 0.649664, 126: 0.710861, 54: 0.740724, 110: 0.654456}

MINUS_INF_1D = (0, 8, 32, 40, 128, 136, 160, 168)

# rule -> (mle, v_front, D_norm, fill), the per-rule means of the 1-D cache.
REFERENCE_1D = {
    150: (1.097203, 1.000000, 0.144129, 0.144129),
    90: (0.691765, 1.000000, 0.068340, 0.068340),
    204: (0.000000, 0.000000, 0.001910, 1.000000),
    170: (0.000000, 1.000000, 0.001910, 0.001910),
    30: (0.660072, 1.000000, 0.315280, 0.315280),
    110: (0.656258, 0.397337, 0.141752, 0.360501),
    122: (0.655884, 0.890527, 0.430524, 0.443179),
    232: (0.693147, 0.003509, 0.003025, 0.528571),
}
SPEARMAN_1D = {"v_front": 0.136886, "D_norm": 0.612437, "fill": 0.290228, "fill_grows": 0.851730}

# label -> (mle, v_front, D_norm, fill), the per-rule means of the 2-D cache:
# the two parity rules (exact ln 5 and ln 4), Life's B3/S23 on the von Neumann
# neighbourhood (a large exponent and next to no damage), the rule with the
# most damage, the null rule and the identity.
REFERENCE_2D = {
    "B13/S024": (1.596990, 1.000000, 0.065695, 0.065695),
    "B13/S13": (1.374072, 1.000000, 0.075015, 0.075015),
    "B3/S23": (1.097976, 0.037534, 0.000554, 0.119787),
    "B13/S02": (1.542056, 1.000000, 0.493192, 0.493192),
    "B/S": (-np.inf, 0.000000, 0.000000, 0.000000),
    "B/S01234": (0.000000, 0.000000, 0.000130, 1.000000),
}
SPEARMAN_2D = {"v_front": 0.561035, "D_norm": 0.516112, "fill": 0.186007, "fill_grows": 0.433686}
MINUS_INF_2D_COUNT = 16


@pytest.fixture(scope="module")
def cache_1d():
    return load(cache_path(1))


@pytest.fixture(scope="module")
def cache_2d():
    return load(cache_path(2))


def per_rule_mle(data):
    mle = data["mle"]
    finite = np.isfinite(mle)
    with np.errstate(invalid="ignore"):
        return np.where(finite.any(axis=1),
                        np.nansum(np.where(finite, mle, np.nan), axis=1) / finite.sum(axis=1),
                        -np.inf)


def row_of(data, key):
    labels = list(data["labels"]) if isinstance(key, str) else [int(r) for r in data["rules"]]
    return labels.index(key)


def spearman(data, key, grows_only=False):
    mle = per_rule_mle(data)
    keep = np.isfinite(mle)
    if grows_only:
        keep &= data["v_front"].mean(axis=1) > CONE_GROWS
    return spearmanr(mle[keep], data[key].mean(axis=1)[keep]).statistic


# --------------------------------------------------------------------------
# The caches are what the figure draws and the manuscript quotes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2])
def test_cache_parameters(dim, cache_1d, cache_2d):
    data = cache_1d if dim == 1 else cache_2d
    d = DEFAULTS[dim]
    assert (int(data["side"]), int(data["T"]), int(data["burn"]), int(data["window"]),
            int(data["samples"]), int(data["seed"])) == (
        d["side"], d["T"], d["burn"], d["window"], d["samples"], SEED)
    assert int(data["side"]) >= 2 * int(data["T"]) + 3
    rules = nonequivalent_ecas() if dim == 1 else nonequivalent_outer_totalistic()
    assert list(data["rules"]) == rules
    assert data["mle"].shape == (len(rules), d["samples"])


def test_1d_reference_values(cache_1d):
    mle = per_rule_mle(cache_1d)
    for rule, (m, v, dn, f) in REFERENCE_1D.items():
        i = row_of(cache_1d, rule)
        assert mle[i] == pytest.approx(m, abs=1e-6)
        assert cache_1d["v_front"][i].mean() == pytest.approx(v, abs=1e-6)
        assert cache_1d["D_norm"][i].mean() == pytest.approx(dn, abs=1e-6)
        assert cache_1d["fill"][i].mean() == pytest.approx(f, abs=1e-6)


def test_1d_spearman(cache_1d):
    assert spearman(cache_1d, "v_front") == pytest.approx(SPEARMAN_1D["v_front"], abs=1e-4)
    assert spearman(cache_1d, "D_norm") == pytest.approx(SPEARMAN_1D["D_norm"], abs=1e-4)
    assert spearman(cache_1d, "fill") == pytest.approx(SPEARMAN_1D["fill"], abs=1e-4)
    assert spearman(cache_1d, "fill", True) == pytest.approx(SPEARMAN_1D["fill_grows"], abs=1e-4)


def test_1d_minus_infinity_rules(cache_1d):
    finite = np.isfinite(cache_1d["mle"])
    dead = [int(r) for r, f in zip(cache_1d["rules"], finite) if not f.any()]
    assert tuple(dead) == MINUS_INF_1D
    # no mixtures: a rule annihilates every sample or none
    assert not np.any(finite.any(axis=1) & ~finite.all(axis=1))
    assert np.all(cache_1d["t_dead"][~finite] >= 1)
    assert np.all(cache_1d["t_dead"][finite] == -1)


def test_1d_affine_rules_match_the_closed_form(cache_1d):
    mle = per_rule_mle(cache_1d)
    for rule in sorted(set(nonequivalent_ecas()) & set(affine_ecas())):
        exact = eca_mle(rule)
        got = mle[row_of(cache_1d, rule)]
        if np.isinf(exact):
            assert got == -np.inf
        else:
            assert abs(got - exact) < 0.01


def test_1d_agrees_with_the_full_qr_cache_of_c10(cache_1d):
    mle = per_rule_mle(cache_1d)
    for rule, ref in FIG6_MLE.items():
        assert abs(mle[row_of(cache_1d, rule)] - ref) < 0.02


def test_1d_front_speed_spans_everything_at_ln2(cache_1d):
    mle = per_rule_mle(cache_1d)
    at_ln2 = np.abs(mle - np.log(2)) < 0.01
    v = cache_1d["v_front"].mean(axis=1)[at_ln2]
    assert v.min() < 0.01 and v.max() > 0.99


def test_1d_damage_of_affine_rules_is_the_defect_pattern(cache_1d):
    # rule 90: 2**popcount(t) damaged cells, radius t, for every sample
    i = row_of(cache_1d, 90)
    T = int(cache_1d["T"])
    t = np.arange(T + 1)
    expected = np.array([2 ** bin(k).count("1") for k in t])
    assert np.array_equal(cache_1d["h_mean"][i], expected)
    assert np.array_equal(cache_1d["r_mean"][i], t)
    assert cache_1d["v_front"][i].std() == 0 and cache_1d["D_norm"][i].std() == 0


def test_2d_reference_values(cache_2d):
    mle = per_rule_mle(cache_2d)
    for label, (m, v, dn, f) in REFERENCE_2D.items():
        i = row_of(cache_2d, label)
        assert mle[i] == pytest.approx(m, abs=1e-6)
        assert cache_2d["v_front"][i].mean() == pytest.approx(v, abs=1e-6)
        assert cache_2d["D_norm"][i].mean() == pytest.approx(dn, abs=1e-6)
        assert cache_2d["fill"][i].mean() == pytest.approx(f, abs=1e-6)


def test_2d_spearman(cache_2d):
    for key, ref in SPEARMAN_2D.items():
        got = spearman(cache_2d, key.replace("_grows", ""), key.endswith("_grows"))
        assert got == pytest.approx(ref, abs=1e-4)


def torus_parity_log_stretch(offsets, v0, T, burn):
    """Exact ln(|J^T v0| / |J^burn v0|) / (T - burn) for the constant parity Jacobian.

    ``J`` is a normal circulant on the torus, diagonalised by the 2-D DFT with
    eigenvalues ``sum_offsets exp(2 pi i (dx k + dy l) / L)``, so the norm of
    ``J^t v0`` follows from the Fourier coefficients of ``v0`` alone.
    """
    L = v0.shape[0]
    k = np.arange(L)
    kk, ll = np.meshgrid(k, k, indexing="ij")
    lam = np.abs(sum(np.exp(2j * np.pi * (dx * kk + dy * ll) / L) for dx, dy in offsets))
    power = np.abs(np.fft.fft2(v0)) ** 2
    norm = lambda t: np.sqrt(np.sum(power * lam ** (2 * t)))
    return (np.log(norm(T)) - np.log(norm(burn))) / (T - burn)


@pytest.mark.parametrize("rule, offsets, exact", [
    (PARITY_VN_INCLUSIVE, VON_NEUMANN_2D, np.log(5)),
    (PARITY_VN_EXCLUSIVE, VON_NEUMANN_2D[1:], np.log(4)),
])
def test_2d_parity_rules_match_the_closed_form(rule, offsets, exact, cache_2d):
    # The single-vector estimate of a constant normal Jacobian is known in
    # closed form for the seeded starting vector, to rounding. Its shortfall
    # from ln 5 / ln 4 is the finite-time transient of the k = 1 estimator at
    # T = 70: the subdominant Fourier modes decay only as (2 pi / L)^2 per
    # step, so 50 window steps leave about 0.01 of it. That is a property of
    # the horizon, not of the code, and is bounded here rather than hidden.
    d = DEFAULTS[2]
    i = row_of(cache_2d, int(rule))
    v0 = tangent_rng(2, int(rule), 0, SEED).standard_normal((d["side"], d["side"]))
    closed_form = torus_parity_log_stretch(offsets, v0, d["T"], d["burn"])
    assert cache_2d["mle"][i, 0] == pytest.approx(closed_form, abs=1e-9)
    shortfall = exact - per_rule_mle(cache_2d)[i]
    assert 0 < shortfall < 0.02


def test_2d_minus_infinity_count(cache_2d):
    finite = np.isfinite(cache_2d["mle"])
    assert int((~finite.any(axis=1)).sum()) == MINUS_INF_2D_COUNT


# --------------------------------------------------------------------------
# The caches are reproducible and the claims are not artefacts of the size
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dim, rule", [(1, 30), (1, 0), (2, PARITY_VN_INCLUSIVE), (2, 392)])
def test_one_sample_recomputes_bit_for_bit(dim, rule, cache_1d, cache_2d):
    data = cache_1d if dim == 1 else cache_2d
    d = DEFAULTS[dim]
    out = run_sample((dim, rule, 0, d["side"], d["T"], d["burn"], d["window"], SEED))
    i = row_of(data, rule)
    for key in ("v_front", "D_norm", "fill", "h_final", "r_final", "mle"):
        assert out[key] == data[key][i, 0]
    assert (out["t_dead"] if out["t_dead"] is not None else -1) == data["t_dead"][i, 0]


def test_small_from_scratch_run_reproduces_the_qualitative_claims():
    small = compute(1, nonequivalent_ecas(), side=61, T=29, burn=9, window=7,
                    samples=2, seed=SEED, workers=1)
    rows = summary_rows(small)
    dead = tuple(r["rule"] for r in rows if r["n_inf"] == r["samples"])
    assert dead == MINUS_INF_1D
    assert spearman(small, "D_norm") > 0.4
    assert spearman(small, "v_front") < 0.4
    # the summary table matches the cache's row layout
    assert set(rows[0]) >= {"rule", "label", "affine", "mle_mean", "mle_sd", "n_inf",
                            "v_front_mean", "D_norm_mean", "fill_mean", "cone_grows"}
