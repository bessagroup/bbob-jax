#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable
from functools import partial
from typing import Optional

# Third-party
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jaxtyping import PRNGKeyArray
from matplotlib.colors import LogNorm, SymLogNorm

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def plot_2d(
    fn: Callable,
    key: PRNGKeyArray,
    bounds: tuple[float, float] = (-5.0, 5.0),
    px: int = 300,
    ax: Optional[plt.Axes] = None,
    log_norm: bool = True,
):
    """
    Plot a 2D heatmap of a function f(x, key) over a given range using imshow.

    Args:
        fn: Callable, function that maps (x, key) -> scalar loss
        key: JAX PRNGKey
        bounds: Tuple (min, max) for x and y axes
        px: Resolution of the mesh grid
        ax: Optional matplotlib Axes
        log_norm: Whether to apply logarithmic normalization to the color scale
    """
    X, Y, Z = create_mesh(fn, key, bounds, px)

    # Create a figure and axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    # Choose normalization
    norm = LogNorm() if log_norm else None

    # Plot with imshow
    _ = ax.imshow(
        Z,
        extent=(*bounds, *bounds),
        origin="lower",
        cmap="viridis",
        norm=norm,
        aspect="auto",
    )

    # Remove ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def plot_3d(
    fn: Callable,
    key: PRNGKeyArray,
    bounds: tuple[float, float] = (-5.0, 5.0),
    px: int = 300,
    ax: Optional[plt.Axes] = None,
):
    X, Y, Z = create_mesh(fn, key, bounds, px)
    Z_shifted = Z - jnp.min(Z)

    # Create a figure and axis if none provided
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Plot the surface
    _ = ax.plot_surface(X, Y, Z_shifted, cmap="viridis",
                        norm=SymLogNorm(), zorder=1)

    # Remove ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig, ax


def create_mesh(
    fn: Callable,
    key: PRNGKeyArray,
    bounds: tuple[float, float],
    px: int,
):
    x_vals = jnp.linspace(*bounds, px)
    X, Y = jnp.meshgrid(x_vals, x_vals)

    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    partial_fn = partial(fn, key=key)
    loss_values = vmap(partial_fn)(points)
    Z = loss_values.reshape(X.shape)

    return X, Y, Z
