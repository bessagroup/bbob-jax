import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray


def fopt(key: PRNGKeyArray) -> jax.Array:
    """Generate a random optimal function value f_opt."""
    return jnp.round(jnp.clip(100.0 * jr.cauchy(key), -1000.0, 1000.0), 2)


def xopt(key: PRNGKeyArray, ndim: int) -> jax.Array:
    """Generate a random optimal solution x_opt within [-4, 4]^ndim."""
    return jr.uniform(key, shape=(ndim,), minval=-5.0, maxval=5.0)


def tosz_func(x):
    def transform(xi):
        c1, c2 = 10., 7.9
        x_sign = jnp.where(xi > 0, 1.0, jnp.where(xi < 0, -1.0, 0.0))
        x_star = jnp.log(jnp.abs(xi))
        return x_sign * jnp.exp(x_star + 0.049 * (jnp.sin(c1 * x_star) + jnp.sin(c2 * x_star)))

    x = jnp.array(x).ravel()
    transformed_x = jnp.where((x == x[0]) | (x == x[-1]), transform(x), x)
    return transformed_x


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
    Q, R_ = jnp.linalg.qr(R)
    # Ensure a right-handed coordinate system (determinant = +1)
    d = jnp.sign(jnp.linalg.det(Q))
    Q = Q * d
    return Q


def penalty(x: jax.Array) -> jax.Array:
    return jnp.sum(
        jnp.power(jnp.maximum(jnp.abs(x) - 5.0, 0.0), 2),
        axis=-1
    )


def bernoulli_vector(dim: int, key: jax.Array) -> jax.Array:
    """Generate a random Bernoulli matrix with entries -1 or 1."""
    return jr.bernoulli(key, p=0.5, shape=(dim,)).astype(jnp.float32) * 2 - 1
