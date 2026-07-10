"""CEC component kernels and hybrid-composition machinery.

Provides the base kernels used inside the CEC 2005 functions
(``ackley``, ``griewank``, ``scaffer_f6``,
``cec2005_weierstrass``) and the CEC 2017 functions
(``cec_bent_cigar``, ``zakharov``, ``cec_rosenbrock``,
``cec_rastrigin``, ``schaffer_f7``, ``levy``,
``modified_schwefel``), plus the differentiable CEC 2005
hybrid composition used by F15-F25 and its ``_f_max``
precomputation. Used by the CEC suites only; the BBOB
transformations live in ``transforms.py``.

Kernel convention: bare ``(ndim,) -> scalar`` functions with
minimum 0 at the origin (exception: ``levy``, whose minimum
is at all-ones — see its docstring). NaN inputs propagate to
the output; ``sqrt`` calls carry a small epsilon instead of
using ``sj.sqrt`` because the latter maps NaN to 0.
"""

#                                                                       Modules
# =============================================================================

# Third-party
from typing import cast

import jax
import jax.numpy as jnp
import softjax as sj

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _finite_like(x: jax.Array) -> jax.Array:
    """Clamp non-finite values to safe finite bounds."""
    finfo = jnp.finfo(x.dtype)
    return jnp.nan_to_num(
        x, nan=0.0, posinf=finfo.max / 1e6, neginf=finfo.min / 1e6
    )


def ackley(x: jax.Array) -> jax.Array:
    """Ackley function. Minimum 0 at origin.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    ndim = x.shape[-1]
    sum_sq = jnp.sum(jnp.square(x))
    sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x))
    # Reformulated to pair cancelling terms and avoid catastrophic
    # cancellation at z=0.  Original: -20*exp(A) - exp(B) + 20 + e
    # Rewrite:  -20*(exp(A) - 1) - (exp(B) - exp(1))
    # Small epsilon in sqrt avoids NaN gradient at x=0 (sqrt'(0) = inf)
    return -20.0 * (jnp.exp(-0.2 * jnp.sqrt(sum_sq / ndim + 1e-20)) - 1.0) - (
        jnp.exp(sum_cos / ndim) - jnp.exp(1.0)
    )


def griewank(x: jax.Array) -> jax.Array:
    """Griewank function. Minimum 0 at origin.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    ndim = x.shape[-1]
    indices = jnp.arange(1, ndim + 1, dtype=x.dtype)
    sum_sq = jnp.sum(jnp.square(x)) / 4000.0
    prod_cos = jnp.prod(jnp.cos(x / jnp.sqrt(indices)))
    return sum_sq - prod_cos + 1.0


def scaffer_f6(x: jax.Array, y: jax.Array) -> jax.Array:
    """Scaffer's F6 2D kernel. Minimum 0 at (0, 0).

    Parameters
    ----------
    x : jax.Array
        First scalar input.
    y : jax.Array
        Second scalar input.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    r2 = jnp.square(x) + jnp.square(y)
    radius = jnp.sqrt(r2 + 1e-12)
    return 0.5 + (jnp.sin(radius) ** 2 - 0.5) / (1.0 + 0.001 * r2) ** 2


def cec2005_weierstrass(x: jax.Array) -> jax.Array:
    """Weierstrass function with CEC 2005 parameters.

    Uses ``a=0.5``, ``b=3``, and 21 terms (k = 0 .. 20).
    Named separately from the BBOB weierstrass which uses
    12 terms with different parameters. Minimum is 0 at
    origin by subtraction of the constant term.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    a, b = 0.5, 3.0
    k = jnp.arange(0, 21, dtype=float)
    ak = a**k  # (21,)
    bk = b**k  # (21,)
    # x: (ndim,), expand to (ndim, 1); k: (21,) → broadcast to (ndim, 21)
    cos_terms = ak * jnp.cos(
        2.0 * jnp.pi * bk * (x[..., None] + 0.5)
    )  # (ndim, 21)
    # Per-element difference avoids catastrophic cancellation at z=0
    cos_ref = ak * jnp.cos(jnp.pi * bk)  # (21,)
    return jnp.sum(cos_terms - cos_ref[None, :])


#                                                             CEC 2017 kernels
# =============================================================================
# Ground truth is the official cec17_test_func.c ("code wins" over the report
# where they disagree); each function in cec2017.py documents its divergences.
# Names clashing with the full BBOB functions in bbob.py carry a ``cec_``
# prefix (cf. ``cec2005_weierstrass``).


def cec_bent_cigar(x: jax.Array) -> jax.Array:
    """Bent Cigar kernel. Minimum 0 at origin.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    return x[0] ** 2 + 1e6 * jnp.sum(x[1:] ** 2)


def zakharov(x: jax.Array) -> jax.Array:
    """Zakharov kernel. Minimum 0 at origin.

    ``sum(x^2) + s^2 + s^4`` with ``s = sum(0.5 * i * x_i)``
    (1-based ``i``). Unimodal.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    ndim = x.shape[-1]
    weights = 0.5 * jnp.arange(1, ndim + 1, dtype=x.dtype)
    s = jnp.sum(weights * x)
    return jnp.sum(jnp.square(x)) + s**2 + s**4


def cec_rosenbrock(x: jax.Array) -> jax.Array:
    """Rosenbrock kernel with the CEC ``+1`` re-centering.

    Evaluates the classic Rosenbrock at ``w = x + 1`` so the
    minimum 0 sits at the origin (the reference code's
    "shift to origin").

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    w = x + 1.0
    return jnp.sum(100.0 * (w[:-1] ** 2 - w[1:]) ** 2 + (w[:-1] - 1.0) ** 2)


def cec_rastrigin(x: jax.Array) -> jax.Array:
    """Rastrigin kernel. Minimum 0 at origin.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    return jnp.sum(jnp.square(x) - 10.0 * jnp.cos(2.0 * jnp.pi * x) + 10.0)


def schaffer_f7(x: jax.Array) -> jax.Array:
    """Schaffer's F7 kernel. Minimum 0 at origin. Needs ``ndim >= 2``.

    ``((1/(D-1)) * sum(sqrt(s_i) * (1 + sin^2(50 * s_i^0.2))))^2``
    with ``s_i = sqrt(x_i^2 + x_{i+1}^2)``. The epsilon inside the
    square roots keeps the gradient finite at the origin while
    letting NaN inputs propagate (unlike ``sj.sqrt``).

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``, ``ndim >= 2``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    ndim = x.shape[-1]
    s = jnp.sqrt(x[:-1] ** 2 + x[1:] ** 2 + 1e-12)
    terms = jnp.sqrt(s) * (1.0 + jnp.sin(50.0 * s**0.2) ** 2)
    return (jnp.sum(terms) / (ndim - 1)) ** 2


def levy(x: jax.Array) -> jax.Array:
    """Levy kernel. Minimum 0 at ``x = 1`` (all-ones), NOT the origin.

    ``w = 1 + (x - 1)/4``; the CEC 2017 reference code applies no
    shrink, so the shifted suite function's argmin is displaced
    from the shift vector by the rotated all-ones point (handled
    by the ``x_opt_from`` resolver in ``spec.py``).

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    w = 1.0 + (x - 1.0) / 4.0
    term1 = jnp.sin(jnp.pi * w[0]) ** 2
    middle = jnp.sum(
        (w[:-1] - 1.0) ** 2
        * (1.0 + 10.0 * jnp.sin(jnp.pi * w[:-1] + 1.0) ** 2)
    )
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + jnp.sin(2.0 * jnp.pi * w[-1]) ** 2)
    return term1 + middle + term3


def modified_schwefel(x: jax.Array) -> jax.Array:
    """Modified Schwefel kernel. Minimum 0 at origin.

    Ported branch-for-branch from ``schwefel_func`` in the
    reference code: ``z = x + 420.9687...``, three regimes
    (``|z| <= 500``, ``z > 500``, ``z < -500``) with an
    out-of-range quadratic penalty scaled by ``1/(10000 D)``.
    All branches are evaluated everywhere and selected with
    ``jnp.where``; their values and gradients are finite for
    finite inputs (epsilon-guarded square roots), so the
    unselected branches contribute exact zeros to the gradient
    and NaN inputs propagate through the selected branch.

    Note
    ----
    The reference code's zero-offset literal ``418.98288...``
    is replaced by evaluating the mid branch at the shift
    constant in the input's dtype (the literal is exactly that
    value in float64), so cancellation at the optimum is exact
    at whichever precision JAX is configured for — the same
    treatment as ``schwefel_xsinx`` in ``bbob.py``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape ``(ndim,)``.

    Returns
    -------
    jax.Array
        Scalar function value.
    """
    ndim = x.shape[-1]

    def mid(z: jax.Array) -> jax.Array:
        return z * jnp.sin(jnp.sqrt(jnp.abs(z) + 1e-12))

    z = x + 4.209687462275036e2
    penalty_scale = 1.0 / (10000.0 * ndim)

    g_mid = mid(z)
    high_arg = 500.0 - jnp.mod(z, 500.0)
    g_high = (
        high_arg * jnp.sin(jnp.sqrt(high_arg + 1e-12))
        - (z - 500.0) ** 2 * penalty_scale
    )
    low_arg = 500.0 - jnp.mod(jnp.abs(z), 500.0)
    g_low = (
        -low_arg * jnp.sin(jnp.sqrt(low_arg + 1e-12))
        - (z + 500.0) ** 2 * penalty_scale
    )

    g = jnp.where(z > 500.0, g_high, jnp.where(z < -500.0, g_low, g_mid))
    g_ref = mid(jnp.asarray(4.209687462275036e2, dtype=x.dtype))
    return g_ref * ndim - jnp.sum(g)


def hybrid_composition(
    x: jax.Array,
    fns: list,
    sigma: jax.Array,
    lambda_: jax.Array,
    bias: jax.Array,
    x_opt: jax.Array,
    M: jax.Array,
    C: float = 2000.0,
    _f_max: jax.Array | None = None,
) -> jax.Array:
    """Hybrid composition kernel for CEC 2005 F15-F25.

    Uses the CEC 2005 composition structure with a smooth
    winner-take-all approximation so the public benchmark
    functions remain compatible with ``jax.grad``.

    Paper structure::

        z = ((x - o_i) / lambda_i) * M_i
        fit_i = f_i(z)
        f_max_i = f_i((y / lambda_i) * M_i), y = [5,...,5]
        fit_i = C * fit_i / f_max_i
        F(x) = sum(w_i * [fit_i + bias_i])

    ``fns`` is a Python list of base functions — always bound
    as a Python constant at construction time, never a
    JAX-traced argument. The Python loop over
    ``num_components`` is unrolled at JIT trace time.

    Parameters
    ----------
    x : jax.Array
        Input point, shape ``(ndim,)``.
    fns : list
        Python list of ``num_components`` base functions.
    sigma : jax.Array
        Basin widths, shape ``(num_components,)``.
    lambda_ : jax.Array
        Per-component input stretch factors,
        shape ``(num_components,)``.
    bias : jax.Array
        Per-component offsets ``[0, 100, ..., 900]``,
        shape ``(num_components,)``.
    x_opt : jax.Array
        Component optima,
        shape ``(num_components, ndim)``.
    M : jax.Array
        Per-component rotation matrices,
        shape ``(num_components, ndim, ndim)``.
    C : float, optional
        Height normalization constant (default 2000).
    _f_max : jax.Array or None, optional
        Precomputed reference normalization values,
        shape ``(num_components,)``. When provided,
        skips per-call reference point evaluation.

    Returns
    -------
    jax.Array
        Scalar composed function value.
    """
    ndim = x.shape[-1]
    num_components = len(fns)

    # --- Weights (CEC 2005 winner-take-all scheme) ---
    diffs = x[None, :] - x_opt
    dist_sq = jnp.sum(diffs**2, axis=-1)
    log_w = -dist_sq / (2.0 * ndim * sigma**2)
    log_w_max = jax.lax.stop_gradient(jnp.max(log_w))
    w = jnp.exp(log_w - log_w_max)
    # SW = sum before suppression (paper divides by this)
    sw = jnp.sum(w)
    w_max = jax.lax.stop_gradient(jnp.max(w))
    # Smooth max indicator for differentiability
    is_max = jax.nn.softmax(1e4 * (w - w_max))
    suppression = 1.0 - sj.clip_st(w_max, 0.0, 1.0) ** 10
    w = w * (is_max + (1.0 - is_max) * suppression)
    w = w / (sw + 1e-30)

    # --- Evaluate each component ---
    def component_value(i: int) -> jax.Array:
        # z = ((x - o_i) / lambda_i) * M_i
        z = M[i] @ ((x - x_opt[i]) / lambda_[i])
        fit = _finite_like(fns[i](z))
        # Height normalize: f_max = f_i((y / lambda_i) * M_i)
        if _f_max is not None:
            f_max = _f_max[i]
        else:
            y = 5.0 * jnp.ones(ndim)
            z_ref = M[i] @ (y / lambda_[i])
            f_max = sj.abs_st(_finite_like(fns[i](z_ref)))
        use_norm = jnp.isfinite(f_max) & (f_max > 1e-30)
        safe_f_max = sj.where(use_norm, f_max, 1.0)
        fit = sj.where(use_norm, C * fit / safe_f_max, fit)
        fit = _finite_like(fit)
        return cast(jax.Array, fit + bias[i])

    values = jnp.stack([component_value(i) for i in range(num_components)])
    return jnp.sum(w * values)


def compute_composition_f_max(
    fns: list,
    lambda_: jax.Array,
    M: jax.Array,
    ndim: int,
) -> jax.Array:
    """Precompute f_max normalization values.

    Parameters
    ----------
    fns : list
        Python list of base component functions.
    lambda_ : jax.Array
        Per-component stretch factors,
        shape ``(num_components,)``.
    M : jax.Array
        Per-component rotation matrices,
        shape ``(num_components, ndim, ndim)``.
    ndim : int
        Number of input dimensions.

    Returns
    -------
    jax.Array
        Reference values of shape ``(num_components,)``.
    """
    y = 5.0 * jnp.ones(ndim)
    f_max_vals = []
    for i in range(len(fns)):
        z_ref = M[i] @ (y / lambda_[i])
        f_max_vals.append(sj.abs_st(_finite_like(fns[i](z_ref))))
    return jnp.stack(f_max_vals)
