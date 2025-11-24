#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable
from typing import Optional

# Third-party
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import PRNGKeyArray
from matplotlib.colors import LogNorm, SymLogNorm

# Local
from .utils import _create_mesh

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
    """Plot a 2D heatmap of a BBOB function.

    Creates a 2D visualization of the function landscape using imshow.

    Parameters
    ----------
    fn : Callable
        BBOB function to plot. Should accept (x, key) parameters.
    key : PRNGKeyArray
        JAX random key for function evaluation.
    bounds : tuple[float, float], optional
        Min and max values for both x and y axes, by default (-5.0, 5.0).
    px : int, optional
        Number of pixels per axis (resolution), by default 300.
    ax : Optional[plt.Axes], optional
        Matplotlib axes to plot on. If None, creates new figure,
        by default None.
    log_norm : bool, optional
        Whether to use logarithmic normalization for colors, by default True.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects containing the plot.
    """

    X, Y, Z = _create_mesh(fn(ndim=2, key=key), bounds, px)

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
    """Plot a 3D surface of a BBOB function.

    Creates a 3D visualization of the function landscape with shifted z-values
    for better visualization.

    Parameters
    ----------
    fn : Callable
        BBOB function to plot. Should accept (x, key) parameters.
    key : PRNGKeyArray
        JAX random key for function evaluation.
    bounds : tuple[float, float], optional
        Min and max values for both x and y axes, by default (-5.0, 5.0).
    px : int, optional
        Number of pixels per axis (resolution), by default 300.
    ax : Optional[plt.Axes], optional
        Matplotlib 3D axes to plot on. If None, creates new figure,
        by default None.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and 3D axes objects containing the plot.
    """
    X, Y, Z = _create_mesh(fn(ndim=2, key=key), bounds, px)
    Z_shifted = Z - jnp.min(Z)

    # Create a figure and axis if none provided
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Plot the surface
    _ = ax.plot_surface(
        X,
        Y,
        Z_shifted,
        cmap="viridis",
        norm=SymLogNorm(
            linthresh=1e-3,
        ),
        zorder=1,
    )

    # Remove ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig, ax
