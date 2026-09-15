"""Shared pytest configuration.

Puts ``scripts/`` (the generators whose caches the claim tests check) and this
directory (so ``test_c12`` can reuse a helper of ``test_c11``) on ``sys.path``
once, whatever the working directory or pytest's import mode.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for folder in (REPO_ROOT / "scripts", REPO_ROOT / "verification"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
