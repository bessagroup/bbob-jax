"""Regenerate the CEC 2017 landscape plots (img/ and docs/img/).

Mirrors update_plots_cec.py for the CEC 2017 suite, with one addition:
functions whose ``min_ndim`` exceeds 2 (the hybrids F11-F20 and the
hybrid-composed F29/F30) cannot be evaluated at ``ndim=2``, so they are
rendered as a **2D slice** of the smallest valid deterministic instance —
the first two coordinates sweep the search range while the remaining
coordinates stay pinned at the optimum (the origin for deterministic
instances). Slice plots carry a "(2D slice of {D}D)" title suffix.
"""

from pathlib import Path

import jax.random as jr
import matplotlib.pyplot as plt

from bbob_jax import cec2017_bounds, cec2017_registry_original
from bbob_jax._src.spec import SPEC_BY_NAME


def _slice_maker(maker, full_ndim: int):
    """Adapt a min_ndim>2 maker to the 2-arg interface plot_2d expects.

    The returned maker builds the deterministic instance at
    ``full_ndim`` and exposes a 2D slice through the optimum plane
    (remaining coordinates pinned at zero, the deterministic optimum).
    """

    def make(ndim, key):
        del ndim
        fn_full, f_opt = maker(ndim=full_ndim, key=key)

        def fn_slice(x2):
            import jax.numpy as jnp

            xx = jnp.zeros(full_ndim, dtype=x2.dtype)
            xx = xx.at[0].set(x2[0]).at[1].set(x2[1])
            return fn_full(xx)

        return fn_slice, f_opt

    return make


def _plot_entries():
    from bbob_jax.plotting import plot_2d, plot_3d

    entries = []
    for name, maker in cec2017_registry_original.items():
        min_ndim = SPEC_BY_NAME[name].min_ndim
        if min_ndim > 2:
            entries.append(
                (
                    name,
                    _slice_maker(maker, min_ndim),
                    f"{name} (2D slice of {min_ndim}D)",
                )
            )
        else:
            entries.append((name, maker, name))
    return entries, plot_2d, plot_3d


def update_plots() -> None:
    Path("img/2d").mkdir(parents=True, exist_ok=True)
    Path("img/3d").mkdir(parents=True, exist_ok=True)

    entries, plot_2d, plot_3d = _plot_entries()

    print("Generating 2D plots...")
    for name, maker, title in entries:
        print(f"Plotting 2D: {name}")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_2d(maker, key=jr.key(0), bounds=cec2017_bounds[name], ax=ax)
        ax.set_title(title)
        plt.savefig(f"img/2d/{name}.png", bbox_inches="tight")
        plt.close(fig)

    print("Generating 3D plots...")
    for name, maker, title in entries:
        print(f"Plotting 3D: {name}")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_3d(maker, key=jr.key(0), bounds=cec2017_bounds[name], ax=ax)
        ax.set_title(title)
        plt.savefig(f"img/3d/{name}.png", bbox_inches="tight")
        plt.close(fig)

    print("Generating 2D Overview Plot...")
    fig, axes = plt.subplots(5, 6, figsize=(24, 20))
    flat = axes.flatten()
    for i, (name, maker, title) in enumerate(entries):
        plot_2d(maker, key=jr.key(0), bounds=cec2017_bounds[name], ax=flat[i])
        flat[i].set_title(title, fontsize=10)
    for j in range(len(entries), len(flat)):
        flat[j].axis("off")
    plt.tight_layout()
    plt.savefig("img/cec2017_functions_overview_2d.png", bbox_inches="tight")
    plt.close(fig)

    print("Generating 3D Overview Plot...")
    fig = plt.figure(figsize=(24, 20))
    for i, (name, maker, title) in enumerate(entries):
        ax = fig.add_subplot(5, 6, i + 1, projection="3d")
        plot_3d(maker, key=jr.key(0), bounds=cec2017_bounds[name], ax=ax)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig("img/cec2017_functions_overview_3d.png", bbox_inches="tight")
    plt.close(fig)

    # docs/img is a symlink to img/ — no sync step needed.
    print("Done.")


if __name__ == "__main__":
    update_plots()
