#!/usr/bin/env python
"""How many workers this machine can actually feed, and what the 88-rule run will cost.

The Benettin pass is one QR factorisation of an N x N matrix per step, so it is
bound by memory bandwidth rather than by arithmetic: past a certain number of
concurrent workers the aggregate throughput stops rising, and adding more only
slows each of them down. That point is a property of the machine, not of the
code, so it has to be measured on the machine that will do the run.

The benchmark times a short Benettin run in each of several worker counts and
reports the aggregate throughput in QR steps per second. Take the worker count
at which throughput stops improving; that is the ``--workers`` to pass to
``scripts/make_nonaffine_spectra.py``. The last column extrapolates the full
88-rule run from the measurement.

On the twelve-core laptop the package was developed on, throughput was 7.0
steps/s on one worker, 12.4 on four, 16.6 on eight and 16.1 on twelve: eight
workers, and an effective speed-up of 2.4 rather than 12.

Usage:
    python scripts/bench_workers.py
    python scripts/bench_workers.py --workers 1 8 16 32 64 --steps 40
"""
from __future__ import annotations

import argparse
import os
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")     # before numpy: one thread per worker

import numpy as np

# The size of the run this benchmark is extrapolating to.
FULL_RULES, FULL_SAMPLES, FULL_STEPS = 79, 40, 500


def _unit(task) -> float:
    """One worker's share: a short Benettin run, timed inside the worker."""
    rule, N, steps, seed = task
    from lyapunov.nonaffine import benettin_log_stretch_trajectory
    state = np.random.default_rng([1, seed]).integers(0, 2, N)
    start = time.perf_counter()
    benettin_log_stretch_trajectory(rule, state, steps)
    return time.perf_counter() - start


def measure(workers: int, N: int, steps: int, rule: int) -> float:
    """Aggregate throughput in QR steps per second at this worker count."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(rule, N, steps, i) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        start = time.perf_counter()
        list(pool.map(_unit, tasks))
        wall = time.perf_counter() - start
    return workers * steps / wall


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--rule", type=int, default=110)
    p.add_argument("--samples", type=int, default=FULL_SAMPLES,
                   help="samples per rule in the run being extrapolated to")
    p.add_argument("--workers", type=int, nargs="+", default=None,
                   help="worker counts to try (default: 1, 2, 4, ... up to the core count)")
    args = p.parse_args(argv)

    cores = os.cpu_count() or 1
    counts = args.workers
    if counts is None:
        counts = [w for w in (1, 2, 4, 8, 16, 32, 64, 128) if w <= cores]
        if cores not in counts:
            counts.append(cores)
    total_steps = FULL_RULES * args.samples * FULL_STEPS

    print(f"{cores} logical cores, N = {args.N}, rule {args.rule}, "
          f"{args.steps} steps per worker")
    print(f"extrapolating to {FULL_RULES} rules x {args.samples} samples x "
          f"{FULL_STEPS} steps = {total_steps:,} QR steps")
    print()
    print(" workers   steps/s   per worker   full run")
    best = (0.0, 0)
    for workers in counts:
        rate = measure(workers, args.N, args.steps, args.rule)
        hours = total_steps / rate / 3600
        print(f"{workers:8d} {rate:9.2f} {rate / workers:12.2f} {hours:9.1f} h",
              flush=True)
        if rate > best[0]:
            best = (rate, workers)
    print()
    print(f"best: {best[1]} workers at {best[0]:.2f} steps/s, "
          f"{total_steps / best[0] / 3600:.1f} h for the Benettin pass")
    print("the exact rank adds about 12 %, and the direct-multiplication "
          "comparison another 17 % unless --skip-direct is passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
