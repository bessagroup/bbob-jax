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
    cec_bent_cigar,
    cec_rastrigin,
    cec_rosenbrock,
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
