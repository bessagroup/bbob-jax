from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import softjax as sj
from jaxtyping import PRNGKeyArray


def _finite_like(x: jax.Array) -> jax.Array:
    finfo = jnp.finfo(x.dtype)
    return jnp.nan_to_num(
        x, nan=0.0, posinf=finfo.max / 1e6, neginf=finfo.min / 1e6
    )


def fopt(key: PRNGKeyArray) -> jax.Array:
    """Generate a random optimal function value f_opt."""
    return jnp.round(
        jnp.clip(100.0 * jr.cauchy(key, shape=()), min=-1000.0, max=1000.0), 2
    )


def xopt(
    key: PRNGKeyArray, ndim: int, minval: float, maxval: float
) -> jax.Array:
    """
    Generate a random optimal solution x_opt within [minval, maxval]^ndim.
    """
    return jr.uniform(key, shape=(ndim,), minval=minval, maxval=maxval)


def tosz_func(x: jax.Array) -> jax.Array:
    c1, c2 = 10.0, 7.9
    eps = 1e-12  # avoid log(0)

    x = jnp.asarray(x)
    abs_x = jnp.maximum(sj.abs_st(x), eps)
    x_sign = sj.sign_st(x)
    x_star = jnp.log(abs_x)
    transformed = x_sign * jnp.exp(
        x_star + 0.049 * (jnp.sin(c1 * x_star) + jnp.sin(c2 * x_star))
    )

    # same “special treatment” as original, but now applied elementwise
    mask = (x == x[0]) | (x == x[-1])
    result: jax.Array = jnp.where(mask, transformed, x)
    return result


def tasy_func(x: jax.Array, beta: float = 0.5) -> jax.Array:
    ndim = x.shape[-1]
    idx = jnp.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim - 1)) * sj.sqrt(jnp.abs(x))
    x_temp = sj.abs_st(x) ** up
    return cast(jax.Array, sj.where(sj.greater_st(x, 0), x_temp, x))


def lambda_func(size: int, alpha: float | jax.Array = 10.0) -> jax.Array:
    idx = jnp.arange(size, dtype=jnp.float32)
    diagonal = alpha ** (idx / (2 * (size - 1)))
    return jnp.diag(diagonal)


def rotation_matrix(dim: int, key: jax.Array) -> jax.Array:
    """Generate a random orthogonal rotation matrix."""
    R = jr.normal(key, shape=(dim, dim))

    # QR decomposition
    orthogonal_matrix, upper_triangular = jnp.linalg.qr(R)

    # Extract diagonal and create sign correction matrix
    diagonal = jnp.diag(upper_triangular)
    sign_correction = jnp.diag(diagonal / jnp.abs(diagonal))

    # Apply sign correction
    rotation = orthogonal_matrix @ sign_correction

    # Ensure determinant is 1 by possibly flipping first row
    determinant = jnp.linalg.det(rotation)
    rotation = rotation.at[0].multiply(determinant)

    return rotation


def penalty(x: jax.Array) -> jax.Array:
    return jnp.sum(jnp.power(sj.relu_st(jnp.abs(x) - 5.0), 2), axis=-1)


def bernoulli_vector(dim: int, key: jax.Array) -> jax.Array:
    """Generate a random Bernoulli matrix with entries -1 or 1."""
    return jr.bernoulli(key, p=0.5, shape=(dim,)).astype(jnp.float32) * 2 - 1


def _create_mesh(
    fn: Callable[[jax.Array], jax.Array],
    bounds: tuple[float, float],
    px: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create a mesh grid and evaluate function values.

    Generates X, Y coordinate meshes and evaluates the function at each point
    to produce Z values.

    Parameters
    ----------
    fn : Callable
        BBOB function to evaluate. Should accept (x,) parameters.
    bounds : tuple[float, float]
        Min and max values for both x and y axes.
    px : int
        Number of pixels per axis (resolution).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        X meshgrid, Y meshgrid, and Z function values.
    """
    x_vals = jnp.linspace(*bounds, px)
    X, Y = jnp.meshgrid(x_vals, x_vals)

    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    loss_values = jax.vmap(fn)(points)
    Z = loss_values.reshape(X.shape)

    return X, Y, Z


def ackley(x: jax.Array) -> jax.Array:
    """Ackley function. Minimum 0 at origin. Accepts 1D input (ndim,)."""
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
    """Griewank function. Minimum 0 at origin. Accepts 1D input (ndim,)."""
    ndim = x.shape[-1]
    indices = jnp.arange(1, ndim + 1, dtype=x.dtype)
    sum_sq = jnp.sum(jnp.square(x)) / 4000.0
    prod_cos = jnp.prod(jnp.cos(x / jnp.sqrt(indices)))
    return sum_sq - prod_cos + 1.0


def scaffer_f6(x: jax.Array, y: jax.Array) -> jax.Array:
    """Scaffer's F6 2D kernel. Minimum 0 at (0, 0). Accepts scalar inputs."""
    r2 = jnp.square(x) + jnp.square(y)
    radius = jnp.sqrt(r2 + 1e-12)
    return 0.5 + (jnp.sin(radius) ** 2 - 0.5) / (1.0 + 0.001 * r2) ** 2


def cec2005_weierstrass(x: jax.Array) -> jax.Array:
    """Weierstrass function with CEC 2005 parameters (a=0.5, b=3, 21 terms).

    Uses k = 0, 1, ..., 20 (jnp.arange(0, 21)). Named separately from the
    existing BBOB weierstrass utility which uses 12 terms with different
    params.
    Minimum is 0 at origin by subtraction of the constant term.
    Accepts 1D input (ndim,).
    """
    a, b = 0.5, 3.0
    k = jnp.arange(0, 21, dtype=jnp.float32)
    ak = a**k  # (21,)
    bk = b**k  # (21,)
    # x: (ndim,), expand to (ndim, 1); k: (21,) → broadcast to (ndim, 21)
    cos_terms = ak * jnp.cos(
        2.0 * jnp.pi * bk * (x[..., None] + 0.5)
    )  # (ndim, 21)
    # Per-element difference avoids catastrophic cancellation at z=0
    cos_ref = ak * jnp.cos(jnp.pi * bk)  # (21,)
    return jnp.sum(cos_terms - cos_ref[None, :])


def hybrid_composition(
    x: jax.Array,
    fns: list,
    sigma: jax.Array,
    lambda_: jax.Array,
    bias: jax.Array,
    x_opt: jax.Array,
    M: jax.Array,
    C: float = 2000.0,
) -> jax.Array:
    """Hybrid composition kernel for CEC 2005 F15-F25.

    Uses the CEC 2005 composition structure with a smooth winner-take-all
    approximation so the public benchmark functions remain compatible with
    ``jax.grad``. The exact paper rule is provided separately in
    ``hybrid_composition_paper_exact``.

    Paper structure:
      z = ((x - o_i) / lambda_i) * M_i
      fit_i = f_i(z)
      f_max_i = f_i((y / lambda_i) * M_i), y = [5,...,5]
      fit_i = C * fit_i / f_max_i
      F(x) = sum(w_i * [fit_i + bias_i])

    fns is a Python list of base functions — always bound as a
    Python constant at construction time, never a JAX-traced
    argument. The Python loop over num_components is unrolled at
    JIT trace time (num_components is fixed).

    Args:
        x:        Input point, shape (ndim,)
        fns:      Python list of num_components base functions
        sigma:    Basin widths, shape (num_components,)
        lambda_:  Per-component input stretch factors,
                  shape (num_components,)
        bias:     Per-component offsets [0, 100, ..., 900],
                  shape (num_components,)
        x_opt:    Component optima, shape (num_components, ndim)
        M:        Per-component rotation matrices,
                  shape (num_c, ndim, ndim)
        C:        Height normalization constant (default 2000).
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
    suppression = 1.0 - jnp.clip(w_max, 0.0, 1.0) ** 10
    w = w * (is_max + (1.0 - is_max) * suppression)
    w = w / (sw + 1e-30)

    # --- Evaluate each component ---
    y = 5.0 * jnp.ones(ndim)

    def component_value(i: int) -> jax.Array:
        # z = ((x - o_i) / lambda_i) * M_i
        z = M[i] @ ((x - x_opt[i]) / lambda_[i])
        fit = _finite_like(fns[i](z))
        # Height normalize: f_max = f_i((y / lambda_i) * M_i)
        z_ref = M[i] @ (y / lambda_[i])
        f_max = jnp.abs(_finite_like(fns[i](z_ref)))
        use_norm = jnp.isfinite(f_max) & (f_max > 1e-30)
        safe_f_max = jnp.where(use_norm, f_max, 1.0)
        fit = jnp.where(use_norm, C * fit / safe_f_max, fit)
        fit = _finite_like(fit)
        return cast(jax.Array, fit + bias[i])

    values = jnp.stack([component_value(i) for i in range(num_components)])
    return jnp.sum(w * values)


def hybrid_composition_paper_exact(
    x: jax.Array,
    fns: list,
    sigma: jax.Array,
    lambda_: jax.Array,
    bias: jax.Array,
    x_opt: jax.Array,
    M: jax.Array,
    C: float = 2000.0,
) -> jax.Array:
    """Reference implementation of the paper's exact composition rule."""
    ndim = x.shape[-1]
    num_components = len(fns)
    diffs = x[None, :] - x_opt
    dist_sq = jnp.sum(diffs**2, axis=-1)
    w = jnp.exp(-dist_sq / (2.0 * ndim * sigma**2))
    sw = jnp.sum(w)
    max_w = jnp.max(w)
    w = jnp.where(w == max_w, w, w * (1.0 - max_w**10))
    w = w / (sw + 1e-30)
    y = 5.0 * jnp.ones(ndim)

    def component_value(i: int) -> jax.Array:
        z = M[i] @ ((x - x_opt[i]) / lambda_[i])
        fit = _finite_like(fns[i](z))
        z_ref = M[i] @ (y / lambda_[i])
        f_max = jnp.abs(_finite_like(fns[i](z_ref)))
        use_norm = jnp.isfinite(f_max) & (f_max > 1e-30)
        safe_f_max = jnp.where(use_norm, f_max, 1.0)
        fit = jnp.where(use_norm, C * fit / safe_f_max, fit)
        fit = _finite_like(fit)
        return cast(jax.Array, fit + bias[i])

    values = jnp.stack([component_value(i) for i in range(num_components)])
    return jnp.sum(w * values)
