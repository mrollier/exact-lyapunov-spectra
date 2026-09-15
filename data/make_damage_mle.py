#!/usr/bin/env python
"""Boolean damage against the maximal Lyapunov exponent, for every rule.

Three catalogues:

  --dim 1                        the 88 ECAs up to reflection and conjugation on a ring
  --dim 2                        the 528 outer-totalistic von Neumann rules up to
                                 conjugation on a torus
  --dim 2 --neighbourhood moore  2000 of the 131 328 outer-totalistic Moore rules up to
                                 conjugation, drawn uniformly (seeded) from the sorted
                                 list of class representatives, on a torus

For each rule and each seeded random initial configuration the same trajectory
yields (i) the damage caused by flipping the central cell -- Hamming distance
``h(t)`` and damage radius ``r(t)``, summarised over the final ``window``
steps as ``v_front``, ``D_norm`` and ``fill`` (see ``lyapunov.damage``) -- and
(ii) the maximal Lyapunov exponent of the Boolean Jacobian from one
renormalised tangent vector. Lattice sizes are primes not smaller than
``2T + 3``, so the damage never wraps and the numbers do not depend on the
lattice at all; the tangent vector is dense and does wrap, and a prime size
keeps it clear of short lattice periods.

Rules that annihilate the tangent vector exactly (rule 0 on the ring, for
instance) have exponent ``-inf``; the cache keeps that value and the step at
which the vector died, and the per-rule table counts such samples in
``n_inf``. A rule's ``mle_mean`` is the mean over its finite samples.

The Moore family is too large to enumerate at these parameters (about 580
CPU-hours), so its catalogue is a sample: ``default_rng([seed, 2, 8])`` with the
repository's seed 20240601 (``SEED``; every configuration and tangent vector is
keyed on it, ``[seed, ..., rule, sample]`` and ``[..., 1]``) draws
``--sample-rules`` classes without replacement from the 131 328 representatives
(``sample_classes``); the sample is stored in the cache and the table. Its
seeds carry the neighbour count, ``[seed, 2, 8, rule, sample]``, so no Moore
sample shares a configuration with a von Neumann one.

Outputs (per dimension ``d``; the Moore run has the suffix ``2d_moore``):
    data/damage/damage_mle_{d}d.npz        per-sample summaries + mean h(t), r(t)
    data/tables/damage_vs_mle_{d}d.csv     one row per rule

Usage:
    python data/make_damage_mle.py --dim 1 --recompute
    python data/make_damage_mle.py --dim 2 --recompute --workers 8
    python data/make_damage_mle.py --dim 2 --neighbourhood moore --recompute   # ~30 min on 10 cores
    python data/make_damage_mle.py --dim 1 --N 41 --T 19 --samples 2 --cache /tmp/smoke.npz
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # before numpy: one thread per worker

import numpy as np

from lyapunov.damage import eca_damage_and_mle, ot_damage_and_mle
from lyapunov.outer_totalistic import MOORE, NEIGHBOURHOODS, bs_notation, nonequivalent_outer_totalistic
from lyapunov.rules import is_affine, nonequivalent_ecas

SEED = 20240601                          # the repository's canonical seed
DEFAULTS = {
    # dim: (side, T, burn, window, samples). Sides are primes >= 2T + 3.
    1: dict(side=607, T=300, burn=100, window=75, samples=24),
    2: dict(side=149, T=70, burn=20, window=17, samples=16),
    # The Moore run keeps the von Neumann protocol and samples the classes.
    "moore": dict(side=149, T=70, burn=20, window=17, samples=10, sample_rules=2000),
}
CONE_GROWS = 0.1                          # v_front above which the cone is said to grow

DATA_DIR = Path(__file__).resolve().parent
TABLE_DIR = DATA_DIR / "tables"


def family(dim: int, neighbourhood: str = "vn"):
    """The key of ``DEFAULTS``: 1, 2 or ``"moore"``."""
    return "moore" if dim == 2 and neighbourhood == "moore" else dim


def _suffix(dim: int, neighbourhood: str = "vn") -> str:
    return f"{dim}d_moore" if family(dim, neighbourhood) == "moore" else f"{dim}d"


def cache_path(dim: int, neighbourhood: str = "vn") -> Path:
    return DATA_DIR / "damage" / f"damage_mle_{_suffix(dim, neighbourhood)}.npz"


def table_path(dim: int, neighbourhood: str = "vn") -> Path:
    return TABLE_DIR / f"damage_vs_mle_{_suffix(dim, neighbourhood)}.csv"


def sample_classes(n: int, seed: int = SEED) -> list[int]:
    """``n`` Moore classes drawn uniformly without replacement, sorted.

    The population is the sorted list of minimal representatives; the draw is
    ``default_rng([seed, 2, 8]).choice``, so the sample is fixed by the seed.
    """
    reps = np.array(nonequivalent_outer_totalistic(MOORE))
    if not 1 <= n <= reps.size:
        raise ValueError(f"Can sample 1..{reps.size} Moore classes, not {n}.")
    chosen = np.random.default_rng([seed, 2, 8]).choice(reps, size=n, replace=False)
    return [int(r) for r in np.sort(chosen)]


def rules_for(dim: int, neighbourhood: str = "vn", sample_rules: int | None = None,
              seed: int = SEED) -> list[int]:
    if family(dim, neighbourhood) == "moore":
        return sample_classes(sample_rules or DEFAULTS["moore"]["sample_rules"], seed)
    return nonequivalent_ecas() if dim == 1 else nonequivalent_outer_totalistic()


def rule_label(dim: int, rule: int, neighbourhood: str = "vn") -> str:
    return str(rule) if dim == 1 else bs_notation(rule, NEIGHBOURHOODS[neighbourhood])


def _seed_key(dim: int, neighbourhood: str, seed: int) -> list[int]:
    if dim == 1:
        return [seed]
    return [seed, 2, 8] if neighbourhood == "moore" else [seed, 2]


def initial_state(dim: int, rule: int, sample: int, side: int, seed: int = SEED,
                  neighbourhood: str = "vn") -> np.ndarray:
    """The seeded random initial configuration of one sample."""
    key = _seed_key(dim, neighbourhood, seed) + [rule, sample]
    shape = side if dim == 1 else (side, side)
    return np.random.default_rng(key).integers(0, 2, size=shape)


def tangent_rng(dim: int, rule: int, sample: int, seed: int = SEED,
                neighbourhood: str = "vn") -> np.random.Generator:
    """The generator that draws the starting tangent vector of one sample."""
    key = _seed_key(dim, neighbourhood, seed) + [rule, sample, 1]
    return np.random.default_rng(key)


def run_sample(task: tuple) -> dict:
    """One rule, one initial configuration: damage summary and exponent.

    ``task`` is ``(dim, rule, sample, side, T, burn, window, seed)`` with an
    optional trailing neighbourhood name (``"vn"`` when absent).
    """
    dim, rule, sample, side, T, burn, window, seed = task[:8]
    neighbourhood = task[8] if len(task) > 8 else "vn"
    state = initial_state(dim, rule, sample, side, seed, neighbourhood)
    rng = tangent_rng(dim, rule, sample, seed, neighbourhood)
    if dim == 1:
        out = eca_damage_and_mle(rule, state, T, burn, window, rng)
    else:
        out = ot_damage_and_mle(rule, state, T, burn, window, rng, NEIGHBOURHOODS[neighbourhood])
    out.update(rule=rule, sample=sample)
    return out


def compute(dim, rules, side, T, burn, window, samples, seed, workers, neighbourhood="vn") -> dict:
    """Every (rule, sample) pair, in a pool of single-threaded workers."""
    R, S = len(rules), samples
    index = {rule: i for i, rule in enumerate(rules)}
    data = {
        "dim": dim, "side": side, "T": T, "burn": burn, "window": window,
        "samples": S, "seed": seed, "rules": np.array(rules),
        "neighbourhood": "ring" if dim == 1 else neighbourhood,
        "labels": np.array([rule_label(dim, r, neighbourhood) for r in rules]),
        "affine": np.array([is_affine(r) if dim == 1 else False for r in rules]),
        "v_front": np.empty((R, S)), "D_norm": np.empty((R, S)), "fill": np.empty((R, S)),
        "h_final": np.empty((R, S), dtype=np.int64), "r_final": np.empty((R, S), dtype=np.int64),
        "mle": np.empty((R, S)), "t_dead": np.full((R, S), -1, dtype=np.int64),
        "h_mean": np.zeros((R, T + 1), dtype=np.int64), "r_mean": np.zeros((R, T + 1), dtype=np.int64),
    }
    tasks = [(dim, rule, s, side, T, burn, window, seed, neighbourhood)
             for rule in rules for s in range(S)]
    t0 = time.time()
    done = 0

    def store(out):
        i = index[out["rule"]]
        s = out["sample"]
        for key in ("v_front", "D_norm", "fill", "h_final", "r_final", "mle"):
            data[key][i, s] = out[key]
        if out["t_dead"] is not None:
            data["t_dead"][i, s] = out["t_dead"]
        data["h_mean"][i] += out["h"]          # integer sums; divided once at the end
        data["r_mean"][i] += out["r"]

    if workers <= 1:
        for task in tasks:
            store(run_sample(task))
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} samples, {time.time() - t0:.0f} s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_sample, task) for task in tasks]
            for fut in as_completed(futures):
                store(fut.result())
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(tasks)} samples, {time.time() - t0:.0f} s", flush=True)
    data["h_mean"] = data["h_mean"] / S
    data["r_mean"] = data["r_mean"] / S
    data["runtime_s"] = time.time() - t0
    return data


def summary_rows(data: dict) -> list[dict]:
    """One row per rule: means and standard deviations over the samples."""
    rows = []
    S = int(data["samples"])
    for i, rule in enumerate(data["rules"]):
        mle = data["mle"][i]
        finite = mle[np.isfinite(mle)]
        n_inf = int(S - finite.size)
        row = {
            "rule": int(rule),
            "label": str(data["labels"][i]),
            "affine": bool(data["affine"][i]),
            "mle_mean": f"{finite.mean():.6f}" if finite.size else "-inf",
            "mle_sd": f"{finite.std(ddof=1):.6f}" if finite.size > 1 else "",
            "n_inf": n_inf,
            "samples": S,
        }
        for key in ("v_front", "D_norm", "fill"):
            vals = data[key][i]
            row[f"{key}_mean"] = f"{vals.mean():.6f}"
            row[f"{key}_sd"] = f"{vals.std(ddof=1):.6f}" if S > 1 else ""
        row["h_final_mean"] = f"{data['h_final'][i].mean():.2f}"
        row["r_final_mean"] = f"{data['r_final'][i].mean():.2f}"
        row["cone_grows"] = bool(data["v_front"][i].mean() > CONE_GROWS)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load(cache: Path) -> dict:
    """Read a cached run back as a plain dictionary."""
    with np.load(cache) as handle:
        return {key: handle[key] for key in handle.files}


def matches(data: dict, side, T, burn, window, samples, seed, rules, neighbourhood="vn") -> bool:
    """True if a cached run was made with exactly these parameters."""
    cached = str(data["neighbourhood"]) if "neighbourhood" in data else "vn"
    return (int(data["side"]) == side and int(data["T"]) == T and int(data["burn"]) == burn
            and int(data["window"]) == window and int(data["samples"]) == samples
            and int(data["seed"]) == seed and list(data["rules"]) == list(rules)
            and (int(data["dim"]) == 1 or cached == neighbourhood))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, choices=(1, 2), required=True)
    p.add_argument("--neighbourhood", choices=("vn", "moore"), default="vn",
                   help="2-D neighbourhood (default von Neumann)")
    p.add_argument("--sample-rules", type=int, default=None,
                   help="Moore only: number of classes drawn (default 2000)")
    p.add_argument("--N", "--L", dest="side", type=int, default=None,
                   help="cells per side (default: 607 for --dim 1, 149 for --dim 2)")
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--burn", type=int, default=None, help="burn-in of the exponent")
    p.add_argument("--window", type=int, default=None, help="final steps the damage is averaged over")
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--cache", default=None)
    p.add_argument("--recompute", action="store_true", help="run even if a matching cache is present")
    p.add_argument("--tables-only", action="store_true",
                   help="rewrite the CSV table from the cache, without computing")
    args = p.parse_args(argv)

    if args.neighbourhood != "vn" and args.dim != 2:
        p.error("--neighbourhood applies to --dim 2 only")
    key = family(args.dim, args.neighbourhood)
    if args.sample_rules is not None and key != "moore":
        p.error("--sample-rules applies to the Moore family only; the others are enumerated")
    defaults = DEFAULTS[key]
    sample_rules = args.sample_rules or defaults.get("sample_rules")
    side = args.side or defaults["side"]
    T = args.T or defaults["T"]
    burn = defaults["burn"] if args.burn is None else args.burn
    window = args.window or defaults["window"]
    samples = args.samples or defaults["samples"]
    if side < 2 * T + 3:
        p.error(f"the side must be at least 2T + 3 = {2 * T + 3} so the damage cannot wrap")
    rules = rules_for(args.dim, args.neighbourhood, sample_rules, args.seed)
    cache = Path(args.cache) if args.cache else cache_path(args.dim, args.neighbourhood)
    # A run at other parameters is a calibration, not the catalogue: keep its
    # table beside its own cache rather than in data/tables.
    default_run = (side, T, burn, window, samples, args.seed, sample_rules) == (
        defaults["side"], defaults["T"], defaults["burn"], defaults["window"],
        defaults["samples"], SEED, defaults.get("sample_rules"))
    table = table_path(args.dim, args.neighbourhood) \
        if default_run and cache == cache_path(args.dim, args.neighbourhood) \
        else cache.with_name(cache.stem + ".csv")

    if args.tables_only:
        data = load(cache)
        print(f"rewriting the table from {cache}")
    else:
        if cache.exists() and not args.recompute:
            data = load(cache)
            if matches(data, side, T, burn, window, samples, args.seed, rules, args.neighbourhood):
                print(f"{cache} already holds this run; pass --recompute to redo it.")
                return 0
            print(f"{cache} was made with different parameters; recomputing.")
        what = f"{len(rules)} sampled Moore classes" if key == "moore" else f"{len(rules)} rules"
        print(f"{what} x {samples} samples, side {side}, T = {T}, "
              f"burn-in {burn}, window {window}, on {args.workers} worker(s)", flush=True)
        data = compute(args.dim, rules, side, T, burn, window, samples, args.seed, args.workers,
                       args.neighbourhood)
        if key == "moore":
            data["sample_rules"] = sample_rules
            data["n_classes"] = len(nonequivalent_outer_totalistic(MOORE))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
        print(f"Wrote {cache} ({cache.stat().st_size / 1e3:.0f} kB) in {data['runtime_s']:.0f} s")

    rows = summary_rows(data)
    _write_csv(rows, table)
    print(f"Wrote {len(rows)} rows to {table}")
    n_inf = sum(r["n_inf"] == r["samples"] for r in rows)
    print(f"  {n_inf} rules with every sample at -inf; "
          f"{sum(0 < r['n_inf'] < r['samples'] for r in rows)} with some")
    finite = [float(r["mle_mean"]) for r in rows if r["mle_mean"] != "-inf"]
    print(f"  finite MLE range {min(finite):.3f} .. {max(finite):.3f}; "
          f"{sum(r['cone_grows'] for r in rows)} rules with a growing cone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
