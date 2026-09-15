"""Exact Lyapunov spectra of affine cellular automata and the parity rule on graphs.

This package is the verified maths core behind the article *"Exact Lyapunov
spectra of affine cellular automata and the parity rule on networks"* (Rollier &
Baetens). The figure scripts and the verification suite both import from the
submodules listed below (the package itself re-exports nothing), so the
mathematics lives in exactly one place.

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
    multiplication at several floating-point precisions (for the benchmark),
    plus the burn-in / starting-frame / checkpoint variant, the unscaled
    direct method of Vispoel et al. (2024) and its float64 precision floor
    (for the convergence notebook), the exact closed-form spectrum they are
    compared against, and the stored per-step log stretches with their
    windowed / cumulative estimators (for the revised Figure 3).
nonaffine
    The other 240 rules: the Boolean Jacobian at a given configuration, the
    banded tangent propagation, Benettin along a trajectory, and the exact rank
    of the tangent map (the number of exponents that are -inf).
outer_totalistic
    The outer-totalistic rules on the 2-D von Neumann neighbourhood (1024
    rules, 528 classes up to conjugation) and on the Moore neighbourhood
    (2**18 rules, 131 328 classes): B/S notation, the torus step, and the
    banded configuration-dependent Boolean Jacobian (five or nine bands).
damage
    Boolean damage from one flipped cell (Hamming distance, damage radius, the
    three light-cone normalisations, diamond or square cone) and the
    single-vector estimate of the maximal Lyapunov exponent, on the same
    trajectory, in one or two dimensions.
parity
    The parity rule on an arbitrary graph: Lyapunov spectrum from the adjacency
    spectrum, defect patterns A^t e_j (mod 2), and (as supplementary material,
    not a claim of the manuscript) eigenvector centrality.
quine_mccluskey
    A small Quine-McCluskey Boolean minimiser used to print gradient entries in
    disjunctive normal form.
vichniac_table1
    Table 1 of Vichniac (1990) exactly as printed, misprints included: the
    single source of truth for the published values.
vichniac
    Recompute all 88 gradients from the definition, compare them by truth table
    with the published table (seven misprinted entries in five rows), and
    generate the manuscript's corrected-entries table.
"""

__version__ = "1.6.0"
