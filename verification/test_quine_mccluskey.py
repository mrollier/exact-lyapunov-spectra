"""Unit tests for the Quine-McCluskey Boolean minimiser (quine_mccluskey.py).

The minimiser turns a truth table into a disjunctive-normal-form expression. Its
one non-negotiable property is *logical equivalence*: the minimised cover must
reproduce the original truth table exactly. We check this exhaustively for all
256 three-variable functions (the setting of the ECA gradient table), and also
that the cover has the fewest products possible, against a brute-force search
over subsets of the prime implicants.
"""
import itertools

import pytest

from lyapunov.quine_mccluskey import (
    _covers,
    minimise_to_implicants,
    implicants_to_truth_table,
    minimise,
    prime_implicants,
)


def _all_truth_tables(n):
    for bits in itertools.product((0, 1), repeat=2 ** n):
        yield list(bits)


def test_equivalence_for_all_three_variable_functions():
    n = 3
    for tt in _all_truth_tables(n):
        impls = minimise_to_implicants(tt, n)
        assert implicants_to_truth_table(impls, n) == tt


def test_constants():
    assert minimise([0, 0, 0, 0, 0, 0, 0, 0], var_names=["a", "b", "c"]) == "0"
    assert minimise([1, 1, 1, 1, 1, 1, 1, 1], var_names=["a", "b", "c"]) == "1"


def test_single_literal():
    # f(a) = a  (n=1): truth table [f(0), f(1)] = [0, 1]
    assert minimise([0, 1], var_names=["a"]) == "a"
    # f(a) = NOT a
    assert minimise([1, 0], var_names=["a"], not_sym="!") == "!a"


def test_or_of_two_literals_is_minimal():
    # f(a, b) = a OR b  -> minterms 01, 10, 11 -> two prime implicants a, b.
    tt = [0, 1, 1, 1]  # index = 2a + b
    impls = minimise_to_implicants(tt, 2)
    assert len(impls) == 2
    s = minimise(tt, var_names=["a", "b"], and_sym="", or_sym=" + ")
    assert s in ("a + b", "b + a")


def test_and_of_two_literals_is_single_term():
    # f(a, b) = a AND b -> single minterm 11 -> one implicant "ab".
    tt = [0, 0, 0, 1]
    impls = minimise_to_implicants(tt, 2)
    assert len(impls) == 1
    assert minimise(tt, var_names=["a", "b"], and_sym="") == "ab"


def _minimum_cover_size(minterms, n):
    primes = prime_implicants(minterms, n)
    for size in range(1, len(primes) + 1):
        for sub in itertools.combinations(primes, size):
            if all(any(_covers(p, m, n) for p in sub) for m in minterms):
                return size
    raise AssertionError("the prime implicants always cover the minterms")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cover_has_the_fewest_products_for_every_function(n):
    # A greedy choice after the essential primes is one product too long for
    # four of the 256 three-variable functions (the cyclic covers of rules 126,
    # 189, 219 and 231); the exhaustive step in the minimiser fixes that.
    for tt in _all_truth_tables(n):
        minterms = [m for m, v in enumerate(tt) if v]
        if not minterms:
            continue
        assert len(minimise_to_implicants(tt, n)) == _minimum_cover_size(minterms, n)


def test_rule_126_phi_has_three_products():
    # rule 126 = 01111110: f = 1 unless all three inputs agree; minimum cover is 3.
    tt = [(126 >> m) & 1 for m in range(8)]
    assert len(minimise_to_implicants(tt, 3)) == 3
