"""Unit tests for ECA rule algebra (src/lyapunov/rules.py).

Conventions used throughout:
- A neighbourhood is (left, centre, right) = (s_{i-1}, s_i, s_{i+1}).
- Wolfram numbering: the output for neighbourhood (l, c, r) is bit (4l + 2c + r)
  of the rule number.
- The Boolean gradient of a rule is the triple of Boolean derivatives
  (d/ds_{i-1}, d/ds_i, d/ds_{i+1}); each derivative is itself a function of the
  neighbourhood, represented as an 8-entry truth table over (l, c, r).
"""
import numpy as np
import pytest

from lyapunov.rules import (
    rule_output,
    gradient_truth_tables,
    is_affine,
    has_constant_jacobian,
    affine_gradient,
    gradient_weight,
    reflect_rule,
    complement_rule,
    affine_ecas,
    nonequivalent_ecas,
)


def test_rule_output_convention_rule90_is_left_xor_right():
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                assert rule_output(90, l, c, r) == (l ^ r)


def test_rule_output_convention_rule150_is_xor_all():
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                assert rule_output(150, l, c, r) == (l ^ c ^ r)


def test_gradient_of_rule30_matches_hand_derivation():
    # Rule 30: phi = s_- XOR (s_o OR s_+).
    #   d/ds_-  = 1
    #   d/ds_o  = NOT s_+     (OR toggles when the other input is 0)
    #   d/ds_+  = NOT s_o
    gtt = gradient_truth_tables(30)  # shape (3, 8), indexed [input, 4l+2c+r]
    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                idx = 4 * l + 2 * c + r
                assert gtt[0, idx] == 1
                assert gtt[1, idx] == (1 - r)
                assert gtt[2, idx] == (1 - c)


def test_affine_rules_are_exactly_the_sixteen():
    expected = {0, 255, 15, 240, 85, 170, 51, 204, 60, 195, 102, 153, 90, 165, 150, 105}
    assert set(affine_ecas()) == expected
    # is_affine agrees with the enumeration for all 256 rules.
    assert {r for r in range(256) if is_affine(r)} == expected


def test_constant_jacobian_iff_affine():
    # Claim C1 in miniature: constant Boolean Jacobian <=> affine.
    for r in range(256):
        assert has_constant_jacobian(r) == is_affine(r)


@pytest.mark.parametrize(
    "rule, grad",
    [
        (0, (0, 0, 0)),
        (15, (1, 0, 0)),
        (85, (0, 0, 1)),
        (51, (0, 1, 0)),
        (60, (1, 1, 0)),
        (102, (0, 1, 1)),
        (90, (1, 0, 1)),
        (150, (1, 1, 1)),
    ],
)
def test_affine_gradients_match_table1(rule, grad):
    assert affine_gradient(rule) == grad
    # A rule and its complement share the gradient (a_0 does not enter J).
    assert affine_gradient(255 - rule) == grad
    assert gradient_weight(rule) == sum(grad)


def test_affine_gradient_raises_for_nonaffine():
    with pytest.raises(ValueError):
        affine_gradient(30)


def test_reflect_and_complement_are_involutions():
    for r in range(256):
        assert reflect_rule(reflect_rule(r)) == r
        assert complement_rule(complement_rule(r)) == r
    # Rule 90 is symmetric under reflection; rule 30 is not self-complementary.
    assert reflect_rule(90) == 90


def test_there_are_88_nonequivalent_ecas():
    reps = nonequivalent_ecas()
    assert len(reps) == 88
    assert len(set(reps)) == 88  # canonical representatives are unique
