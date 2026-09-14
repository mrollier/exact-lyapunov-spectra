#!/usr/bin/env python
"""Execute the notebooks in place and fail if any cell fails.

Usage:
    python notebooks/execute.py                       # every notebook here, in order
    python notebooks/execute.py 04_convergence_figure # one of them, extension optional

Requires the ``notebook`` extra (``pip install -e .[notebook]``: nbformat,
nbclient, ipykernel). ``reproduce.py all`` calls this script; each notebook's
final cell asserts its reported numbers (the convergence study against an
independent reimplementation, ``fig3_convergence_study.py``; the figure notebook
against the reference table) and raises on the first discrepancy, so a non-zero
exit here means a real mismatch, not a missing dependency (that case is reported
separately and exits zero, so a machine without Jupyter can still reproduce the
rest of the package).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ["03_benettin_convergence.ipynb", "04_convergence_figure.ipynb"]


def main() -> int:
    # Pin the BLAS thread count for the kernel: faster on these small matrices
    # and byte-identical numbers between runs (see the notebook's first cell).
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[skip] notebook execution needs the 'notebook' extra ({exc}); "
              "install with: pip install -e .[notebook]")
        return 0
    wanted = [a if a.endswith(".ipynb") else a + ".ipynb" for a in sys.argv[1:]] or NOTEBOOKS
    for name in wanted:
        path = HERE / name
        if not path.exists():
            print(f"[error] no such notebook: {path}")
            return 1
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(HERE)}})
        t0 = time.time()
        try:
            client.execute()
        finally:
            nbformat.write(nb, path)  # keep outputs even on failure, for inspection
        print(f"Executed {path.name} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
