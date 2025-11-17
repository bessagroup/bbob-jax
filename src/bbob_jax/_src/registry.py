from collections.abc import Callable
from functools import wraps

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from bbob_jax._src.utils import fopt, rotation_matrix, xopt

BBOBFn = Callable[[jax.Array, PRNGKeyArray], jax.Array]

registry_original: dict[str, BBOBFn] = {}

registry: dict[str, BBOBFn] = {}


def register_function(format: str) -> Callable[[BBOBFn], BBOBFn]:
    def decorator(fn: BBOBFn) -> BBOBFn:
        @wraps(fn)
        def wrapper_det(x: jax.Array, *args, **kwargs) -> jax.Array:
            ndim = x.shape[-1]
            x_opt = jnp.zeros(ndim)
            eye = jnp.eye(ndim)
            return fn(x, x_opt=x_opt, R=eye, Q=eye)

        @wraps(fn)
        def wrapper_rand(
            x: jax.Array, key: PRNGKeyArray, *args, **kwargs
        ) -> jax.Array:
            ndim = x.shape[-1]
            key1, key2 = jr.split(key)
            x_opt = xopt(key, ndim)
            R = rotation_matrix(ndim, key1)
            Q = rotation_matrix(ndim, key2)
            return fn(x, x_opt=x_opt, R=R, Q=Q) + fopt(key)

        # Register both variants
        registry_original[format] = wrapper_det
        registry[format] = wrapper_rand

        return wrapper_rand  # return original function (not wrapped)

    return decorator
