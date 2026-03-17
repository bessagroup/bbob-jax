from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray


def fopt(key: PRNGKeyArray) -> jax.Array:
    """Generate a random optimal function value f_opt."""
    return jnp.round(
        jnp.clip(100.0 * jr.cauchy(key, shape=()), min=-1000.0, max=1000.0), 2
    )


def xopt(key: PRNGKeyArray, ndim: int) -> jax.Array:
    """Generate a random optimal solution x_opt within [-4, 4]^ndim."""
    return jr.uniform(key, shape=(ndim,), minval=-4.0, maxval=4.0)


def tosz_func(x: jax.Array) -> jax.Array:
    c1, c2 = 10.0, 7.9
    eps = 1e-12  # avoid log(0)

    x = jnp.asarray(x)
    abs_x = jnp.maximum(jnp.abs(x), eps)
    x_sign = jnp.sign(x)
    x_star = jnp.log(abs_x)
    transformed = x_sign * jnp.exp(
        x_star + 0.049 * (jnp.sin(c1 * x_star) + jnp.sin(c2 * x_star))
    )

    # same “special treatment” as original, but now applied elementwise
    mask = (x == x[0]) | (x == x[-1])
    return jnp.where(mask, transformed, x)


def tasy_func(x: jax.Array, beta: float = 0.5) -> jax.Array:
    ndim = x.shape[-1]
    idx = jnp.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim - 1)) * jnp.sqrt(jnp.abs(x))
    x_temp = jnp.abs(x) ** up
    return cast(jax.Array, jnp.where(x > 0, x_temp, x))


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
    return jnp.sum(jnp.power(jnp.maximum(jnp.abs(x) - 5.0, 0.0), 2), axis=-1)


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
    sum_sq = jnp.sum(x**2)
    sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x))
    return (
        -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum_sq / ndim))
        - jnp.exp(sum_cos / ndim)
        + 20.0
        + jnp.e
    )


def griewank(x: jax.Array) -> jax.Array:
    """Griewank function. Minimum 0 at origin. Accepts 1D input (ndim,)."""
    ndim = x.shape[-1]
    indices = jnp.arange(1, ndim + 1, dtype=jnp.float32)
    sum_sq = jnp.sum(x**2) / 4000.0
    prod_cos = jnp.prod(jnp.cos(x / jnp.sqrt(indices)))
    return sum_sq - prod_cos + 1.0


def scaffer_f6(x: jax.Array, y: jax.Array) -> jax.Array:
    """Scaffer's F6 2D kernel. Minimum 0 at (0, 0). Accepts scalar inputs."""
    r2 = x**2 + y**2
    return 0.5 + (jnp.sin(jnp.sqrt(r2)) ** 2 - 0.5) / (1.0 + 0.001 * r2) ** 2


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
    outer_sum = jnp.sum(cos_terms)
    # Constant subtraction: n * sum_k(ak * cos(2*pi*bk*0.5))
    ndim = x.shape[-1]
    constant = ndim * jnp.sum(ak * jnp.cos(jnp.pi * bk))
    return outer_sum - constant


def hybrid_composition(
    x: jax.Array,
    fns: list,
    sigma: jax.Array,
    lambda_: jax.Array,
    bias: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Hybrid composition kernel for CEC 2005 F15-F25.

    Applies double rotation per component: z_rot = Q[i] @ (R[i] @ z),
    matching the CEC 2005 spec's use of two rotation matrices per component.
    Callers that only need one rotation should pass Q=identity stack (F15).

    fns is a Python list of base functions — always bound as a Python constant
    at construction time, never a JAX-traced argument. The Python loop over
    num_components is unrolled at JIT trace time (num_components is fixed).

    Args:
        x:        Input point, shape (ndim,)
        fns:      Python list of num_components base functions (ndim,)->scalar
        sigma:    Basin widths, shape (num_components,)
        lambda_:  Per-component scaling (includes height normalization),
                  shape (num_components,)
        bias:     Per-component offsets [0, 100, 200, ..., 900],
                  shape (num_components,)
        x_opt:    Component optima, shape (num_components, ndim)
        R:        Per-component first rotation matrices,
                  shape (num_c, ndim, ndim)
        Q:        Per-component second rotation matrices,
                  shape (num_c, ndim, ndim)
    """
    ndim = x.shape[-1]
    num_components = len(fns)  # Python constant — unrolled at trace time

    # Unnormalized weights
    diffs = x[None, :] - x_opt  # (num_components, ndim)
    dist_sq = jnp.sum(diffs**2, axis=-1)  # (num_components,)
    w = jnp.exp(-dist_sq / (2.0 * ndim * sigma**2))  # (num_components,)

    # Safe normalization: if all weights underflow to 0, use first component
    w_sum = jnp.sum(w)
    first_only = jnp.zeros(num_components).at[0].set(1.0)
    w = jnp.where(w_sum > 0.0, w / w_sum, first_only)

    # Evaluate each component (Python loop — unrolled at JIT trace time)
    # Double rotation: Q[i] @ (R[i] @ z) per CEC 2005 spec
    def component_value(i: int) -> jax.Array:
        z = (x - x_opt[i]) / sigma[i]
        z_rot = Q[i] @ (R[i] @ z)
        return cast(jax.Array, lambda_[i] * fns[i](z_rot) + bias[i])

    values = jnp.stack([component_value(i) for i in range(num_components)])
    return jnp.sum(w * values)
