"""Random parameter sampling shared by the BBOB and CEC 2005 factories.

Provides the samplers for optimal values (``fopt``), optimal
locations (``xopt``), sign vectors (``bernoulli_vector``) and
orthogonal rotation matrices (``rotation_matrix``).
"""

#                                                                       Modules
# =============================================================================

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def fopt(key: PRNGKeyArray) -> jax.Array:
    """Generate a random optimal function value f_opt.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.

    Returns
    -------
    jax.Array
        Scalar optimal function value clipped to [-1000, 1000].
    """
    return jnp.round(
        jnp.clip(100.0 * jr.cauchy(key, shape=()), min=-1000.0, max=1000.0), 2
    )


def xopt(
    key: PRNGKeyArray, ndim: int, minval: float, maxval: float
) -> jax.Array:
    """Generate a random optimal solution x_opt.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    ndim : int
        Number of dimensions.
    minval : float
        Lower bound for each coordinate.
    maxval : float
        Upper bound for each coordinate.

    Returns
    -------
    jax.Array
        Random point of shape ``(ndim,)`` in
        ``[minval, maxval]^ndim``.
    """
    return jr.uniform(key, shape=(ndim,), minval=minval, maxval=maxval)


def rotation_matrix(dim: int, key: jax.Array) -> jax.Array:
    """Generate a random orthogonal rotation matrix.

    Parameters
    ----------
    dim : int
        Matrix dimension.
    key : jax.Array
        JAX random key.

    Returns
    -------
    jax.Array
        Orthogonal matrix of shape ``(dim, dim)`` with
        determinant 1.
    """
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


def bernoulli_vector(dim: int, key: jax.Array) -> jax.Array:
    """Generate a random Bernoulli vector with entries -1 or 1.

    Parameters
    ----------
    dim : int
        Length of the vector.
    key : jax.Array
        JAX random key.

    Returns
    -------
    jax.Array
        Vector of shape ``(dim,)`` with entries in {-1, 1}.
    """
    return jr.bernoulli(key, p=0.5, shape=(dim,)).astype(float) * 2 - 1
