"""Input/output transformations from the BBOB suite.

Provides the smooth log-sine deformation (``tosz_func``), the
asymmetry transformation (``tasy_func``), the diagonal
conditioning matrix (``lambda_func``) and the boundary
``penalty``. Used by the BBOB function implementations only;
the CEC 2005 suite has its own kernels in ``composition.py``.
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


def tosz_func(x: jax.Array) -> jax.Array:
    """Smooth log-sine deformation (T_osz) from the BBOB suite."""
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
    """Asymmetry transformation (T_asy) from the BBOB suite."""
    ndim = x.shape[-1]
    idx = jnp.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim - 1)) * sj.sqrt(jnp.abs(x))
    x_temp = sj.abs_st(x) ** up
    return cast(jax.Array, sj.where(sj.greater_st(x, 0), x_temp, x))


def lambda_func(size: int, alpha: float | jax.Array = 10.0) -> jax.Array:
    """Diagonal conditioning matrix (Lambda) from the BBOB suite."""
    idx = jnp.arange(size, dtype=float)
    diagonal = alpha ** (idx / (2 * (size - 1)))
    return jnp.diag(diagonal)


def penalty(x: jax.Array) -> jax.Array:
    """Boundary penalty: squared excess beyond [-5, 5]."""
    return jnp.sum(jnp.power(sj.relu_st(jnp.abs(x) - 5.0), 2), axis=-1)
