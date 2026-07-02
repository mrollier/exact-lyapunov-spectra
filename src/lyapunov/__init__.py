"""Exact Lyapunov spectra of affine cellular automata and the parity rule on graphs.

This package is the verified maths core behind the article *"Exact Lyapunov
spectra of affine cellular automata and the parity rule on networks"* (Rollier &
Baetens). The figure scripts and the verification suite both import from here, so
the mathematics lives in exactly one place.

Modules
-------
gf2
    Integer- and GF(2)-safe matrix powers. Avoids the silent int64 overflow of
    ``numpy.linalg.matrix_power`` that corrupts A^t computations.
rules
    Elementary cellular automaton (ECA) rule tables, Boolean gradients (the
    Vichniac derivative), affine/constant-Jacobian detection, and enumeration of
    the 88 non-equivalent ECAs.
jacobian
    Constant Boolean Jacobians: the circulant Jacobian of an affine ECA and the
    adjacency Jacobian of the parity rule, plus a pure-numpy ECA step.
spectra
    Closed-form singular values / Lyapunov spectra via the discrete Fourier
    transform of the gradient stencil, in any dimension, and the neighbourhood
    structure factor.
benettin
    Reference numerical Lyapunov routines: Benettin's QR algorithm and direct
    multiplication at several floating-point precisions (for the benchmark).
parity
    The parity rule on an arbitrary graph: Lyapunov spectrum from the adjacency
    spectrum, defect patterns A^t e_j (mod 2), and eigenvector centrality.
quine_mccluskey
    A small Quine-McCluskey Boolean minimiser used to print gradient entries in
    disjunctive normal form.
"""

__version__ = "1.0.0"
