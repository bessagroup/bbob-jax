import jax
import jax.numpy as jnp

from .utils import diag_func, tasy_func, tosz_func


# F01
def sphere(x: jax.Array) -> jax.Array:
    return jnp.sum(x ** 2)


# F02
def ellipsoidal(x: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    idx = jnp.arange(ndim, dtype=x.dtype)
    z = tosz_func(x)
    weights = 10.0 ** (6.0 * idx / (ndim - 1))
    return jnp.sum(weights * z**2)

# F03


def rastrigin(x: jax.Array) -> jax.Array:
    ndim = x.shape[-1]

    alpha = diag_func(ndim, alpha=10.0)
    temp = tosz_func(x)
    z = jnp.matmul(alpha, tasy_func(temp, beta=0.2))

    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0 * ndim)


# F04
def f(x):
    ndim = x.shape[-1]
    d = jnp.arange(1, ndim, dtype=x.dtype)
    s = 10 ** (0.5 * (d - 1) / (ndim - 1))
    z = s * tosz_func(x)
    first_term = jnp.sum(z**2)

    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0 * ndim)
