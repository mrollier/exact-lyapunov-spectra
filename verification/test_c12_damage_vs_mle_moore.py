"""Claim C12 -- Boolean damage against the maximal exponent for the sampled
outer-totalistic Moore rules (bottom panel of Fig. 7).

The Moore family ("Life-like" rules) has 2**18 = 262 144 rules in 131 328
classes under conjugation, too many to enumerate at the parameters of the
von Neumann panel, so the panel is a uniform sample of 2000 classes drawn
with a fixed seed. These tests pin the sample itself (it must be exactly the
one ``sample_classes`` regenerates from the seed, all distinct minimal
representatives), the parameters of the committed cache, a few per-rule
numbers, Spearman's rho, and the -inf convention (15 rules with every sample
annihilated and, new to this family, 8 rules with only some); they tie the estimator to
the closed form on the two Moore parity rules (exponents ln 9 and ln 8, whose
Jacobian is the constant 3 x 3 torus adjacency) and pin Life's single-vector
exponent, both computed on the fly because the sample is purely random; and
they recompute one cached sample bit for bit and run the whole pipeline from
scratch at a small size, so a stale cache cannot make this file pass.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from lyapunov.outer_totalistic import (
    MOORE, PARITY_MOORE_EXCLUSIVE, PARITY_MOORE_INCLUSIVE, conjugate_rule, ot_from_bs)
from lyapunov.spectra import MOORE_2D

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from make_damage_mle import (  # noqa: E402
    DEFAULTS, SEED, cache_path, compute, load, run_sample, sample_classes, summary_rows,
    tangent_rng)
from test_c11_damage_vs_mle import torus_parity_log_stretch  # noqa: E402

N_CLASSES = 131328
LIFE = ot_from_bs("B3/S23", MOORE)

# label -> (mle, v_front, D_norm, fill), per-rule means over the 10 samples of
# the committed cache, for a handful of the sampled classes.
REFERENCE_MOORE = {
    "B05/S": (1.089874, 0.204206, 0.004672, 0.107627),            # first sampled class
    "B03568/S01367": (1.817461, 1.000000, 0.471305, 0.471305),
    "B/S1345678": (1.378591, 0.024346, 0.000228, 0.227075),      # last sampled class
    "B246/S01246": (2.028097, 1.000000, 0.498102, 0.498102),     # the most damage
    "B0246/S0135": (2.147185, 1.000000, 0.497893, 0.497893),     # the largest exponent
    "B458/S": (-np.inf, 0.000000, 0.000000, 0.000000),           # every sample annihilated
    "B67/S045": (0.000000, 0.025969, 0.000078, 0.043356),        # exponent exactly 0
    "B46/S45": (0.693147, 0.000000, 0.000000, 0.000000),         # 9 of 10 samples annihilated
}
SPEARMAN_MOORE = {"v_front": 0.840436, "D_norm": 0.822012, "fill": 0.759376, "fill_grows": 0.807932}
MINUS_INF_MOORE_COUNT = 15
# Unlike the ring and the von Neumann torus, the Moore sample has rules whose
# samples are only partly annihilated: the configuration usually dies to a
# fixed point with a zero Jacobian, but some initial configurations leave a
# small surviving pattern. Their exponent is the mean over the finite samples.
MIXED_MOORE = {"B46/S45": 9, "B4578/S56": 3, "B7/S37": 3, "B367/S57": 7, "B36/S457": 4,
               "B367/S68": 8, "B5678/S4568": 8, "B368/S678": 9}     # label -> dead samples
# Single-vector exponents of sample 0 (seeded tangent vector) of the anchor
# rules, which are not in the random sample and are recomputed here.
LIFE_MLE_SAMPLE0 = 1.341872
PARITY_SHORTFALL_MAX = 0.02     # ln 9 / ln 8 minus the finite-horizon estimate (0.013, 0.015)


@pytest.fixture(scope="module")
def cache():
    return load(cache_path(2, "moore"))


def per_rule_mle(data):
    mle = data["mle"]
    finite = np.isfinite(mle)
    with np.errstate(invalid="ignore"):
        return np.where(finite.any(axis=1),
                        np.nansum(np.where(finite, mle, np.nan), axis=1) / finite.sum(axis=1),
                        -np.inf)


def row_of(data, label):
    return int(np.flatnonzero(data["labels"] == label)[0])


def spearman(data, key, grows_only=False):
    x = per_rule_mle(data)
    y = data[key].mean(axis=1)
    keep = np.isfinite(x)
    if grows_only:
        keep &= data["v_front"].mean(axis=1) > 0.1
    return float(spearmanr(x[keep], y[keep]).statistic)


# --------------------------------------------------------------------------
# The sample and the cache
# --------------------------------------------------------------------------

def test_the_sample_is_the_seeded_draw_of_distinct_representatives(cache):
    rules = [int(r) for r in cache["rules"]]
    assert rules == sample_classes(2000, SEED)
    assert len(set(rules)) == 2000 and rules == sorted(rules)
    for rule in rules[:200] + rules[-200:]:
        assert rule <= conjugate_rule(rule, MOORE)
    assert int(cache["n_classes"]) == N_CLASSES and int(cache["sample_rules"]) == 2000
    with pytest.raises(ValueError):
        sample_classes(N_CLASSES + 1)
    # A different seed gives a different sample; the same seed the same one.
    assert sample_classes(2000, SEED + 1) != rules
    assert sample_classes(50, SEED) != sample_classes(50, SEED)[::-1]


def test_cache_parameters(cache):
    d = DEFAULTS["moore"]
    assert str(cache["neighbourhood"]) == "moore" and int(cache["dim"]) == 2
    assert int(cache["side"]) == d["side"] == 149
    assert (int(cache["T"]), int(cache["burn"]), int(cache["window"])) == (70, 20, 17)
    assert int(cache["samples"]) == d["samples"] == 10
    assert int(cache["seed"]) == SEED
    assert cache["mle"].shape == (2000, 10) and cache["h_mean"].shape == (2000, 71)


def test_reference_values(cache):
    mle = per_rule_mle(cache)
    for label, (m, v, D, f) in REFERENCE_MOORE.items():
        i = row_of(cache, label)
        assert mle[i] == pytest.approx(m, abs=1e-6)
        assert cache["v_front"][i].mean() == pytest.approx(v, abs=1e-6)
        assert cache["D_norm"][i].mean() == pytest.approx(D, abs=1e-6)
        assert cache["fill"][i].mean() == pytest.approx(f, abs=1e-6)


def test_spearman(cache):
    for key, ref in SPEARMAN_MOORE.items():
        got = spearman(cache, key.replace("_grows", ""), key.endswith("_grows"))
        assert got == pytest.approx(ref, abs=1e-4)


def test_minus_infinity_and_mixed_rules(cache):
    finite = np.isfinite(cache["mle"])
    dead = ~finite.any(axis=1)
    assert int(dead.sum()) == MINUS_INF_MOORE_COUNT
    assert np.all(cache["t_dead"][dead] >= 1)
    assert np.all(cache["D_norm"][dead] == 0)
    mixed = finite.any(axis=1) & ~finite.all(axis=1)
    got = {str(cache["labels"][i]): int((~finite[i]).sum()) for i in np.flatnonzero(mixed)}
    assert got == MIXED_MOORE
    # A dead sample has a recorded death step and a finite one has none.
    assert np.all((cache["t_dead"] >= 1) == ~finite)


# --------------------------------------------------------------------------
# Anchors computed on the fly: the parity rules and Life
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule, offsets, exact", [
    (PARITY_MOORE_INCLUSIVE, MOORE_2D, np.log(9)),
    (PARITY_MOORE_EXCLUSIVE, tuple(o for o in MOORE_2D if o != (0, 0)), np.log(8)),
])
def test_moore_parity_rules_match_the_closed_form(rule, offsets, exact):
    # The single-vector estimate of a constant normal Jacobian is known in
    # closed form for the seeded starting vector, to rounding; what remains is
    # the finite-horizon transient, bounded here (see the C11 note).
    d = DEFAULTS["moore"]
    out = run_sample((2, rule, 0, d["side"], d["T"], d["burn"], d["window"], SEED, "moore"))
    v0 = tangent_rng(2, rule, 0, SEED, "moore").standard_normal((d["side"], d["side"]))
    closed_form = torus_parity_log_stretch(offsets, v0, d["T"], d["burn"])
    assert out["mle"] == pytest.approx(closed_form, abs=1e-9)
    assert 0 < exact - out["mle"] < PARITY_SHORTFALL_MAX
    assert out["v_front"] == pytest.approx(1.0)


def test_life_sample_is_pinned():
    d = DEFAULTS["moore"]
    out = run_sample((2, LIFE, 0, d["side"], d["T"], d["burn"], d["window"], SEED, "moore"))
    assert out["mle"] == pytest.approx(LIFE_MLE_SAMPLE0, abs=1e-6)
    assert out["t_dead"] is None


# --------------------------------------------------------------------------
# The cache is reproducible and the pipeline runs from scratch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("position", [0, 1234, 1999])
def test_one_sample_recomputes_bit_for_bit(position, cache):
    d = DEFAULTS["moore"]
    rule = int(cache["rules"][position])
    out = run_sample((2, rule, 3, d["side"], d["T"], d["burn"], d["window"], SEED, "moore"))
    for key in ("v_front", "D_norm", "fill", "h_final", "r_final", "mle"):
        assert out[key] == cache[key][position, 3]
    assert (out["t_dead"] if out["t_dead"] is not None else -1) == cache["t_dead"][position, 3]


def test_small_from_scratch_run():
    rules = sample_classes(20, SEED)
    small = compute(2, rules, side=31, T=14, burn=4, window=4, samples=2, seed=SEED,
                    workers=1, neighbourhood="moore")
    rows = summary_rows(small)
    assert len(rows) == 20 and [r["rule"] for r in rows] == rules
    assert str(small["neighbourhood"]) == "moore"
    assert all("/" in r["label"] and set(r["label"]) <= set("BS/012345678") for r in rows)
    assert np.all(small["h_final"] <= (2 * 14 + 1) ** 2)
    assert np.all(small["D_norm"] <= 1)
