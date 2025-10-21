from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jaxtyping import PRNGKeyArray
from matplotlib.colors import LogNorm


def plot_3d(
    fn: Callable,
    key: PRNGKeyArray,
    bounds: tuple[float, float] = (-5.0, 5.0),
    px: int = 300,
    ax: Optional[plt.Axes] = None,
):
    x_vals = jnp.linspace(*bounds, px)
    X, Y = jnp.meshgrid(x_vals, x_vals)

    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    partial_fn = partial(fn, key=key)
    loss_values = vmap(partial_fn)(points)
    Z = loss_values.reshape(X.shape)

    # Create a figure and axis if none provided
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Plot the surface
    _ = ax.plot_surface(X, Y, Z, cmap="viridis", norm=LogNorm(), zorder=1)

    # Remove ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig, ax
    return fig, ax
