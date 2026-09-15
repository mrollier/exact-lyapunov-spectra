#!/usr/bin/env python
"""Launcher for ``scripts/verify_vichniac.py``, kept at the repository root.

Appendix A of the manuscript invites the reader to run ``python
verify_vichniac.py`` in this repository, so that command has to keep working
from the root. The script itself lives with the other generators in
``scripts/``; this file only hands over to it, with the same arguments, and
returns its exit code.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "verify_vichniac.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
