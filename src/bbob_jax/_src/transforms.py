"""Input/output transformations from the BBOB suite.

Provides the smooth log-sine deformation (``tosz_func``), the
asymmetry transformation (``tasy_func``), the diagonal
conditioning matrix (``lambda_func``) and the boundary
``penalty``. Used by the BBOB noiseless and BBOB-noisy function
implementations; the CEC 2005 suite has its own kernels in
``composition.py``.

``tosz_func`` and ``tasy_func`` replicate the official reference
code (``benchmarkshelper.c``: ``monotoneTFosc``; the asymmetric
transform inlined per function) exactly; they are cross-checked
against the compiled legacy C through both suites (see
``scripts/crosscheck_bbob_noiseless.py`` and
``scripts/crosscheck_bbob_noisy.py``). Before ADR 0005 both
carried deviations (first/last-element mask, single-branch
constants, off-by-one exponent) — values produced by older
versions are not comparable.
"""

#                                                                       Modules
# =============================================================================

# Third-party
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
    """Smooth log-sine deformation (T_osz) from the BBOB suite.

    Applied elementwise to every component, with the
    sign-dependent constants of the reference: ``(10, 7.9)`` for
    positive inputs, ``(5.5, 3.1)`` for negative ones. Zero maps
    to zero.

    Parameters
    ----------
    x : jax.Array
        Input array (any shape, applied elementwise).

    Returns
    -------
    jax.Array
        Deformed array of the same shape.
    """
    eps = 1e-12  # avoid log(0) in the unselected branch
    x_hat = jnp.log(jnp.maximum(sj.abs_st(x), eps))
    pos = jnp.exp(
        x_hat + 0.049 * (jnp.sin(10.0 * x_hat) + jnp.sin(7.9 * x_hat))
    )
    neg = -jnp.exp(
        x_hat + 0.049 * (jnp.sin(5.5 * x_hat) + jnp.sin(3.1 * x_hat))
    )
    return jnp.where(x > 0, pos, jnp.where(x < 0, neg, x))


def tasy_func(x: jax.Array, beta: float = 0.5) -> jax.Array:
    """Asymmetry transformation (T_asy) from the BBOB suite.

    Positive components are raised to
    ``1 + beta * (i / (ndim - 1)) * sqrt(x_i)`` (0-based ``i``),
    negative ones pass through — the reference's exponent
    (``beta * linspace(0, 1, ndim)``).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    beta : float
        Asymmetry strength.

    Returns
    -------
    jax.Array
        Transformed array of the same shape.
    """
    ndim = x.shape[-1]
    idx = jnp.arange(ndim, dtype=x.dtype)
    x_safe = jnp.maximum(x, 0.0)
    exponent = 1.0 + beta * (idx / max(ndim - 1, 1)) * sj.sqrt(x_safe)
    return jnp.where(x > 0, (x_safe + 1e-99) ** exponent, x)


def lambda_func(size: int, alpha: float | jax.Array = 10.0) -> jax.Array:
    """Diagonal conditioning matrix (Lambda) from the BBOB suite."""
    idx = jnp.arange(size, dtype=float)
    diagonal = alpha ** (idx / (2 * (size - 1)))
    return jnp.diag(diagonal)


def penalty(x: jax.Array) -> jax.Array:
    """Boundary penalty: squared excess beyond [-5, 5]."""
    return jnp.sum(jnp.power(sj.relu_st(jnp.abs(x) - 5.0), 2), axis=-1)
