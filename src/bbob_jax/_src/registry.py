from functools import wraps
from typing import Callable

import jax

BBOBFn = Callable[[jax.Array], jax.Array]

registry: dict[str, BBOBFn] = {

}


def register_function(format: str) -> Callable[[BBOBFn], BBOBFn]:
    def decorator(fn: BBOBFn) -> BBOBFn:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> None:
            return fn(*args, **kwargs)
        # Here you can add the function to a registry based on the format
        # For example:
        registry[format] = fn
        return wrapper
    return decorator
