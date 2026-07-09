"""Mesh-grid evaluation of 2D benchmark functions.

Used by the plotting module and the vmap tests; kept free of
matplotlib so it can be imported without the optional ``plot``
dependency group.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

# Third-party
import jax
import jax.numpy as jnp

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


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
