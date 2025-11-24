from collections.abc import Callable

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


def tosz_func(x):
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


def tasy_func(x: jax.Array, beta=0.5) -> jax.Array:
    ndim = x.shape[-1]
    idx = jnp.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim - 1)) * jnp.sqrt(jnp.abs(x))
    x_temp = jnp.abs(x) ** up
    return jnp.where(x > 0, x_temp, x)


def lambda_func(size: int, alpha: float = 10.0) -> jax.Array:
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
    fn: Callable,
    bounds: tuple[float, float],
    px: int,
):
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
