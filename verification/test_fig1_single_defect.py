"""Fig. 1 starts from a single defect: one flipped cell at t = 0 and no re-injection."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lyapunov.gf2 import gf2_matrix_power
from lyapunov.jacobian import eca_jacobian

FIGURES = Path(__file__).resolve().parent.parent / "figures"


@pytest.fixture(scope="module")
def fig():
    import sys
    sys.path.insert(0, str(FIGURES))  # the figure script imports _style
    spec = importlib.util.spec_from_file_location("fig_defect_cones", FIGURES / "fig_defect_cones.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rule", [90, 150])
@pytest.mark.parametrize("seed", [0, 3])
def test_linear_panels_are_the_exact_power_of_the_jacobian(fig, rule, seed):
    N, T = fig.N_DEFAULT, 40
    pattern = fig.difference_pattern(rule, N, T, seed)
    J = np.asarray(eca_jacobian(rule, N), dtype=int)
    for t in range(T + 1):
        assert np.array_equal(pattern[t], gf2_matrix_power(J, t)[:, N // 2])


def test_published_panels_start_from_one_cell_and_survive(fig):
    for rule in fig.CLASS_IV + fig.CLASS_III:
        pattern = fig.difference_pattern(rule, fig.N_DEFAULT, fig.T_DEFAULT, fig.SEED_DEFAULT)
        assert pattern[0].sum() == 1 and pattern[0, fig.N_DEFAULT // 2] == 1
        assert pattern[-1].any()
