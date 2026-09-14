# Running the 88-rule catalogue on the workstation

Disposable working note. Delete it once the run is done and its numbers have
moved into `docs/provenance.md` and the manuscript.

The goal: Lyapunov spectra of **all 88 ECAs up to reflection and conjugation**,
at Vispoel's settings (N = 1000, periodic, T = 500, burn-in 200, window 300, 40
random initial configurations per rule). Nine of the 88 are affine and come from
the closed form in milliseconds; the other **79 need trajectories** and are the
whole cost of the exercise.

Everything below is one command. No code needs editing.

---

## 0. Before anything: do not run this inside OneDrive

The run rewrites a 40 MB checkpoint every 100 samples for several hours. Inside
a synced folder that is thousands of uploads and a good chance of a corrupted
file. **Clone to a local disk** (`C:\work\`, `/scratch/`, `~/runs/`, whatever the
workstation calls it) and work there.

```bash
git clone https://github.com/mrollier/exact-lyapunov-spectra.git
cd exact-lyapunov-spectra
git checkout main && git pull
```

## 1. Environment

CPython 3.10 or newer (3.11 is what everything was tested on).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Confirm the machine agrees with the committed results before spending hours on
it. This takes about 90 s and must end with every test passing:

```bash
python -m pytest -q
```

Then a two-minute smoke test of the exact pipeline the long run uses, at a size
that finishes immediately:

```bash
python data/make_nonaffine_spectra.py --all-88 --recompute \
    --N 40 --T 20 --burn 10 --samples 2 --workers 4 \
    --cache /tmp/smoke88.npz
```

It should print 88 rules and write `/tmp/smoke88_spectra.csv` beside the cache.
If that works, the full command differs only in its numbers.

## 2. Find the right worker count. This is the one decision that matters

The bottleneck is **memory bandwidth, not cores**. Each step is one QR
factorisation of a 1000 x 1000 matrix, which streams about 8 MB; past some
number of concurrent workers the machine cannot feed them and throughput stops
rising. On the twelve-core laptop this was developed on, throughput was 7.0
steps/s on one worker, 12.4 on four, 16.6 on eight and 16.1 on twelve. Eight
workers, an effective speed-up of 2.4 rather than 12. **Do not assume a 64-core
machine gives 64x.** Measure it:

```bash
python data/bench_workers.py
```

It prints aggregate throughput per worker count and extrapolates the full run.
Take the worker count where throughput stops improving; going past it makes the
run slower, not faster. That number is `--workers` below. The whole benchmark
takes a couple of minutes.

If the machine has a lot of cores, extend the range:

```bash
python data/bench_workers.py --workers 1 8 16 32 48 64 --steps 40
```

## 3. The run

```bash
python data/make_nonaffine_spectra.py --all-88 --recompute --workers <from step 2>
```

Start it under `screen`, `tmux` or `nohup` so an ssh drop does not kill it:

```bash
nohup python data/make_nonaffine_spectra.py --all-88 --recompute \
      --workers 32 > run88.log 2>&1 &
tail -f run88.log
```

It prints a running count and an estimate of the time left.

**If it dies, resume it.** A checkpoint is written every 100 samples to
`data/nonaffine/spectra_all88.partial.npz`:

```bash
python data/make_nonaffine_spectra.py --all-88 --recompute --resume --workers 32
```

Only the missing samples are recomputed. Every sample is seeded from
`(20240601, rule, sample)` alone, so a resumed run is **bitwise identical** to an
uninterrupted one, at any worker count, in any order. Two tests in
`verification/test_nonaffine.py` pin exactly that. The checkpoint is deleted when
the run completes.

### What it costs

Measured per sample at N = 1000, T = 500, on one core: 56 s for the Benettin
pass, 7.6 s for the exact rank, 11 s for the direct-multiplication comparison.
79 rules x 40 samples is therefore about **81 core-hours**, or 67 with
`--skip-direct`. Divide by the effective speed-up from step 2, which is the
number that decides whether this is three hours or thirty.

`--skip-direct` drops the table showing that Vispoel's own method cannot produce
these spectra. Worth keeping if the catalogue is going into the paper; worth
dropping if this is a first look.

### Useful variants

```bash
# a cheaper catalogue: N = 500 costs one eighth as much (the cost is cubic in N)
python data/make_nonaffine_spectra.py --all-88 --recompute --N 500 \
       --cache data/nonaffine/spectra_all88_N500.npz

# fewer samples: the spread is small, 16 samples still gives the mean to ~0.006
python data/make_nonaffine_spectra.py --all-88 --recompute --samples 16

# rewrite the CSVs from a finished cache without recomputing anything
python data/make_nonaffine_spectra.py --all-88 --tables-only
```

## 4. What comes out

| file | what it is |
|---|---|
| `data/nonaffine/spectra_all88.npz` | every per-sample spectrum and rank, about 30 MB |
| `data/nonaffine/all88_spectra.csv` | 88 rows: MLE, spread, share of the spectrum at -infinity |
| `data/nonaffine/all88_direct_multiplication.csv` | what Vispoel's method returns for each |

The `.npz` files are **git-ignored on purpose** (30 MB is too much for the
repository). Bring the two CSVs back in git and the `.npz` back by hand, or add
it with git-lfs if it is worth versioning.

Nothing here touches the Figure 6 cache (`data/nonaffine/spectra.npz`), the
manuscript tables in `data/tables/`, or any committed number. The two runs are
independent by construction.

## 5. Expect some rules to have no spectrum at all

Several of the 88 annihilate the tangent space completely: their Boolean
Jacobian is the zero matrix at every configuration reached, so **every** exponent
is `-inf`, the rank is zero and `mle_mean` is `-inf` with `nan` for the spread.
Rule 0 is the obvious one; at the smoke-test size rules 8 and 32 did it too.
This is a correct answer, not a failure, but it means the catalogue cannot be
plotted or tabulated as if every rule had a finite maximal exponent. Decide how
to present those rows before building the figure.

## 6. Afterwards

- The CSVs are the deliverable. A 2 x 6 histogram figure does not scale to 88
  panels; a table of 88 rows plus one summary plot (rules ranked by MLE, or MLE
  against the annihilated share) is the presentation that works.
- Record the run in `docs/provenance.md` the way the nine-rule run is recorded:
  command, expected output, wall-clock, machine.
- Update `CHANGELOG.md` and delete this file.
