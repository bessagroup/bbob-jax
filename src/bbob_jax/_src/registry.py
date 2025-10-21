from functools import wraps
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from bbob_jax._src.utils import fopt, xopt

BBOBFn = Callable[[jax.Array, PRNGKeyArray], jax.Array]

registry_original: dict[str, BBOBFn] = {

}

registry: dict[str, BBOBFn] = {

}


def register_function(format: str) -> Callable[[BBOBFn], BBOBFn]:
    def decorator(fn: BBOBFn) -> BBOBFn:
        @wraps(fn)
        def wrapper_det(x: jax.Array, *args, key: PRNGKeyArray, **kwargs) -> jax.Array:
            x_opt = jnp.zeros(shape=x.shape[-1])
            return fn(x, *args, key=key, x_opt=x_opt, **kwargs)

        @wraps(fn)
        def wrapper_rand(x: jax.Array, *args, key: PRNGKeyArray, **kwargs) -> jax.Array:
            x_opt = xopt(key, ndim=x.shape[-1])
            return fn(x, *args, key=key, x_opt=x_opt, **kwargs) + fopt(key)

        # Register both variants
        registry_original[format] = wrapper_det
        registry[format] = wrapper_rand

        return wrapper_rand  # return original function (not wrapped)
    return decorator
