"""CEC 2017 benchmark functions implemented in JAX.

IMPORTANT: This implementation targets parity with the official reference
code ``cec17_test_func.c`` (Awad et al., 2016, "Problem Definitions and
Evaluation Criteria for the CEC 2017 Special Session and Competition on
Single Objective Real-Parameter Numerical Optimization"). Where the report
and the reference code disagree, the code wins — that is what published
results were produced with — and the divergence is documented in the
function's docstring. Parameters (shift vectors, rotation matrices) are
generated from seeds rather than loaded from the official data files, and
the per-function bias values ``F_i* = 100 * i`` are replaced by a sampled
``f_opt``. Results will NOT match published CEC 2017 benchmarking results.

F2 (Sum of Different Powers) was officially withdrawn from the competition
because of unstable behavior at higher dimensions and is not implemented;
the numbering keeps the hole (``f1``, ``f3`` .. ``f30``), matching the
reference code and data files. The "final version updated" report renumbers
the remaining functions 1-29 — that renumbering is NOT followed here.

All functions share the internal signature ``fn(x, x_opt, f_opt, R, Q)``
(``Q`` is unused by the simple functions F1-F10) and are exposed through
``cec2017_registry`` / ``cec2017_registry_original`` after partial
application, with search range ``[-100, 100]^D`` and shifts sampled in
``[-80, 80]^D`` as in the official suite.
"""

import math

import jax
import jax.numpy as jnp

from bbob_jax._src.composition import (
    ackley,
    cec2005_weierstrass,
    cec2017_hybrid_partition,
    cec2017_katsuura,
    cec_bent_cigar,
    cec_rastrigin,
    cec_rosenbrock,
    discus,
    expanded_griewank_rosenbrock,
    expanded_schaffer_f6,
    hgbat,
    high_conditioned_elliptic,
    levy,
    modified_schwefel,
    schaffer_f7,
    zakharov,
)

__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"


def f1(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Bent Cigar (F1).

    Unimodal, non-separable, smooth but narrow ridge.
    ``z = R @ (x - x_opt)``, no shrink.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return cec_bent_cigar(z) + f_opt


def f3(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Zakharov (F3).

    Unimodal, non-separable. ``z = R @ (x - x_opt)``, no shrink.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return zakharov(z) + f_opt


def f4(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Rosenbrock (F4).

    Multimodal (for D > 3), non-separable.
    ``z = R @ (2.048/100 * (x - x_opt))`` with the kernel's
    ``+1`` re-centering so the minimum stays at ``x_opt``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ ((2.048 / 100.0) * (x - x_opt))
    return cec_rosenbrock(z) + f_opt


def f5(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Rastrigin (F5).

    Multimodal, non-separable, huge number of local optima.
    ``z = R @ (5.12/100 * (x - x_opt))``. The report's equation
    omits the shrink; the reference code applies ``5.12/100``
    (code wins).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ ((5.12 / 100.0) * (x - x_opt))
    return cec_rastrigin(z) + f_opt


def f6(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Schaffer's F7 (F6). Needs ``ndim >= 2``.

    Multimodal, asymmetrical. The report defines this slot as
    ``f_schafferF7(M(0.5/100 * (x - o)))``, but in the reference
    code the kernel reads the pre-rotation global ``y = x - o``
    with shrink rate 1.0 — the rotation matrix is computed and
    then never used, and no shrink is applied. Code wins:
    this function is shifted only (``rotated=False`` tag), with
    no shrink. ``R`` and ``Q`` are accepted but unused.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused; dead in the reference code).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = x - x_opt
    return schaffer_f7(y) + f_opt


def f7(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Lunacek Bi-Rastrigin (F7).

    Multimodal, asymmetrical, two funnels. Ported from
    ``bi_rastrigin_func`` in the reference code: ``y = 10/100 *
    (x - x_opt)``, doubled and sign-flipped where the *shift*
    coordinate is negative (a static, per-instance flip), the
    two funnel terms are computed on the unrotated variable and
    only the cosine term sees the rotation. The report's outer
    ``600/100`` scale (eq. 25) double-counts the basic
    function's internal ``10/100``; the code applies ``10/100``
    once (code wins). The ``min`` of the funnel terms uses
    ``jnp.minimum`` (well-defined subgradient).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (applied to the cosine term only).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(ndim + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)

    y = (10.0 / 100.0) * (x - x_opt)
    flip = jnp.where(x_opt < 0.0, -1.0, 1.0)
    t = 2.0 * flip * y
    funnel0 = jnp.sum(t**2)
    funnel1 = d * ndim + s * jnp.sum((t + (mu0 - mu1)) ** 2)
    t_rot = R @ t
    cos_term = 10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * t_rot)))
    return jnp.minimum(funnel0, funnel1) + cos_term + f_opt


def f8(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated "Non-Continuous" Rastrigin (F8).

    Multimodal, non-separable. The report's basic definition
    includes a non-continuity rounding step plus ``tosz``/
    ``tasy``/conditioning transforms, but in the reference code
    the rounding loop operates on a stale global buffer *before*
    it is overwritten and none of the transforms appear — the
    function actually computed (and used for all published
    results) is plain shifted-rotated Rastrigin,
    ``z = R @ (5.12/100 * (x - x_opt))``, identical in structure
    to F5. Code wins; only the sampled instance differs from F5.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ ((5.12 / 100.0) * (x - x_opt))
    return cec_rastrigin(z) + f_opt


def f9(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Levy (F9).

    Multimodal, huge number of local optima. ``z = R @ (x -
    x_opt)``; the report says shrink ``5.12/100`` but the
    reference code applies none (code wins). NOTE: the Levy
    kernel's minimum is at all-ones, not the origin, so the
    global minimizer is ``x_opt + R.T @ ones`` — NOT the sampled
    shift (the same is true of the official suite, where
    ``F9(o_9) != 900``). The ``x_opt_from`` resolver in
    ``spec.py`` accounts for this; the minimizer can fall
    outside ``[-100, 100]`` only for shifts within ``sqrt(D)``
    of the boundary (shifts are sampled in ``[-80, 80]``).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return levy(z) + f_opt


def f10(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted and Rotated Modified Schwefel (F10).

    Multimodal, huge number of local optima, second-best
    optimum far from the global one.
    ``z = R @ (1000/100 * (x - x_opt))``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (10.0 * (x - x_opt))
    return modified_schwefel(z) + f_opt


#                                                     Hybrid functions F11-F20
# =============================================================================
# Shared structure (hf01-hf10 in the reference code): the whole input is
# shifted, rotated and then SHUFFLED (``y = (R @ (x - x_opt))[shuffle]``),
# split into contiguous chunks by the official proportions, and each chunk
# is fed to one kernel with a kernel-specific shrink rate. Chunk sizes come
# from :func:`cec2017_hybrid_partition` (the reference ceil rule wherever it
# is well-defined). The trailing ``0.0 * jnp.sum(y)`` term is an exact zero
# for finite inputs and exists solely to propagate NaN through coordinates
# a hybrid ignores (single-coordinate Rosenbrock chunks have an empty sum;
# the Schaffer F7 sub-kernel reads the wrong coordinates entirely — see
# ``f14``/``f20``).


def _hybrid_chunks(
    y: jax.Array,
    proportions: tuple[float, ...],
    min_sizes: tuple[int, ...],
) -> list[jax.Array]:
    """Split the shuffled vector into per-kernel chunks."""
    sizes = cec2017_hybrid_partition(y.shape[-1], proportions, min_sizes)
    chunks = []
    start = 0
    for size in sizes:
        chunks.append(y[start : start + size])
        start += size
    return chunks


def _hybrid_lunacek(chunk: jax.Array, shift_prefix: jax.Array) -> jax.Array:
    """Lunacek Bi-Rastrigin as a hybrid sub-kernel (``bi_rastrigin_func``
    with ``s_flag = r_flag = 0``).

    The sign flip reads the LEADING entries of the hybrid's full,
    unshuffled shift vector — not the shift belonging to this chunk's
    coordinates. That is what the reference code's global-buffer
    aliasing computes, and what all published results used; replicated
    faithfully. No rotation is applied to the cosine term.

    Needs a chunk of at least two dimensions: at one dimension the
    depth factor ``s = 1 - 1/(2 sqrt(D+20) - 8.2)`` turns negative
    and ``mu1`` is the square root of a negative number (NaN in the
    reference code as well).
    """
    ndim = chunk.shape[-1]
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(ndim + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)

    y = (10.0 / 100.0) * chunk
    flip = jnp.where(shift_prefix < 0.0, -1.0, 1.0)
    t = 2.0 * flip * y
    funnel0 = jnp.sum(t**2)
    funnel1 = d * ndim + s * jnp.sum((t + (mu0 - mu1)) ** 2)
    cos_term = 10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * t)))
    return jnp.minimum(funnel0, funnel1) + cos_term


def f11(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 1 (F11): Zakharov, Rosenbrock, Rastrigin.

    Proportions (0.2, 0.4, 0.4); needs ``ndim >= 3``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3 = _hybrid_chunks(y, (0.2, 0.4, 0.4), (1, 1, 1))
    return (
        zakharov(c1)
        + cec_rosenbrock((2.048 / 100.0) * c2)
        + cec_rastrigin((5.12 / 100.0) * c3)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f12(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 2 (F12): Elliptic, Modified Schwefel, Bent Cigar.

    Proportions (0.3, 0.3, 0.4); needs ``ndim >= 3``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3 = _hybrid_chunks(y, (0.3, 0.3, 0.4), (1, 1, 1))
    return (
        high_conditioned_elliptic(c1)
        + modified_schwefel(10.0 * c2)
        + cec_bent_cigar(c3)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f13(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 3 (F13): Bent Cigar, Rosenbrock, Lunacek
    Bi-Rastrigin.

    Proportions (0.3, 0.3, 0.4); needs ``ndim >= 4`` (the Lunacek
    chunk must hold at least two dimensions — see
    ``_hybrid_lunacek``). The Lunacek sub-kernel's sign flip reads
    the leading entries of the full shift vector (reference-code
    global-buffer aliasing).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3 = _hybrid_chunks(y, (0.3, 0.3, 0.4), (1, 1, 2))
    return (
        cec_bent_cigar(c1)
        + cec_rosenbrock((2.048 / 100.0) * c2)
        + _hybrid_lunacek(c3, x_opt[: c3.shape[-1]])
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f14(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 4 (F14): Elliptic, Ackley, Schaffer F7,
    Rastrigin.

    Proportions (0.2, 0.2, 0.2, 0.4); needs ``ndim >= 6`` (the
    Schaffer F7 chunk must hold at least two dimensions). In the
    reference code the Schaffer F7 sub-kernel reads the LEADING
    entries of the shuffled vector instead of its own chunk (stale
    global-buffer aliasing) — its own chunk's coordinates never
    influence the value. Replicated faithfully (published results
    used it); the ``0.0 * sum(y)`` term keeps NaN propagating
    through the ignored coordinates.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4 = _hybrid_chunks(y, (0.2, 0.2, 0.2, 0.4), (1, 1, 2, 1))
    return (
        high_conditioned_elliptic(c1)
        + ackley(c2)
        + schaffer_f7(y[: c3.shape[-1]])
        + cec_rastrigin((5.12 / 100.0) * c4)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f15(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 5 (F15): Bent Cigar, HGBat, Rastrigin,
    Rosenbrock.

    Proportions (0.2, 0.2, 0.3, 0.3); needs ``ndim >= 4``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4 = _hybrid_chunks(y, (0.2, 0.2, 0.3, 0.3), (1, 1, 1, 1))
    return (
        cec_bent_cigar(c1)
        + hgbat((5.0 / 100.0) * c2)
        + cec_rastrigin((5.12 / 100.0) * c3)
        + cec_rosenbrock((2.048 / 100.0) * c4)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f16(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 6 (F16): Expanded Schaffer F6, HGBat,
    Rosenbrock, Modified Schwefel.

    Proportions (0.2, 0.2, 0.3, 0.3); needs ``ndim >= 4``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4 = _hybrid_chunks(y, (0.2, 0.2, 0.3, 0.3), (1, 1, 1, 1))
    return (
        expanded_schaffer_f6(c1)
        + hgbat((5.0 / 100.0) * c2)
        + cec_rosenbrock((2.048 / 100.0) * c3)
        + modified_schwefel(10.0 * c4)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f17(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 7 (F17): Katsuura, Ackley, Expanded
    Griewank-Rosenbrock, Modified Schwefel, Rastrigin.

    Proportions (0.1, 0.2, 0.2, 0.2, 0.3); needs ``ndim >= 5``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4, c5 = _hybrid_chunks(
        y, (0.1, 0.2, 0.2, 0.2, 0.3), (1, 1, 1, 1, 1)
    )
    return (
        cec2017_katsuura((5.0 / 100.0) * c1)
        + ackley(c2)
        + expanded_griewank_rosenbrock((5.0 / 100.0) * c3)
        + modified_schwefel(10.0 * c4)
        + cec_rastrigin((5.12 / 100.0) * c5)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f18(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 8 (F18): Elliptic, Ackley, Rastrigin, HGBat,
    Discus.

    Proportions (0.2, 0.2, 0.2, 0.2, 0.2); needs ``ndim >= 5``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4, c5 = _hybrid_chunks(
        y, (0.2, 0.2, 0.2, 0.2, 0.2), (1, 1, 1, 1, 1)
    )
    return (
        high_conditioned_elliptic(c1)
        + ackley(c2)
        + cec_rastrigin((5.12 / 100.0) * c3)
        + hgbat((5.0 / 100.0) * c4)
        + discus(c5)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f19(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 9 (F19): Bent Cigar, Rastrigin, Expanded
    Griewank-Rosenbrock, Weierstrass, Expanded Schaffer F6.

    Proportions (0.2, 0.2, 0.2, 0.2, 0.2); needs ``ndim >= 5``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4, c5 = _hybrid_chunks(
        y, (0.2, 0.2, 0.2, 0.2, 0.2), (1, 1, 1, 1, 1)
    )
    return (
        cec_bent_cigar(c1)
        + cec_rastrigin((5.12 / 100.0) * c2)
        + expanded_griewank_rosenbrock((5.0 / 100.0) * c3)
        + cec2005_weierstrass((0.5 / 100.0) * c4)
        + expanded_schaffer_f6(c5)
        + 0.0 * jnp.sum(y)
        + f_opt
    )


def f20(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _shuffle: jax.Array,
) -> jax.Array:
    """Hybrid Function 10 (F20): HGBat, Katsuura, Ackley, Rastrigin,
    Modified Schwefel, Schaffer F7.

    Proportions (0.1, 0.1, 0.2, 0.2, 0.2, 0.2); needs ``ndim >= 7``
    (the reference ceil split is ill-defined below ``ndim = 10``;
    the repair split covers 7-9, keeping the Schaffer F7 chunk at
    two dimensions). As in ``f14``, the Schaffer F7 sub-kernel reads
    the LEADING entries of the shuffled vector instead of its own
    chunk — reference-code global-buffer aliasing, replicated
    faithfully.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).
    _shuffle : jax.Array
        Dimension permutation (identity when deterministic).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    y = (R @ (x - x_opt))[_shuffle]
    c1, c2, c3, c4, c5, c6 = _hybrid_chunks(
        y, (0.1, 0.1, 0.2, 0.2, 0.2, 0.2), (1, 1, 1, 1, 1, 2)
    )
    return (
        hgbat((5.0 / 100.0) * c1)
        + cec2017_katsuura((5.0 / 100.0) * c2)
        + ackley(c3)
        + cec_rastrigin((5.12 / 100.0) * c4)
        + modified_schwefel(10.0 * c5)
        + schaffer_f7(y[: c6.shape[-1]])
        + 0.0 * jnp.sum(y)
        + f_opt
    )
