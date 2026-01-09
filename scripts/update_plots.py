from pathlib import Path

import jax.random as jr
import matplotlib.pyplot as plt

from bbob_jax import registry_original
from bbob_jax.plotting import plot_2d, plot_3d


def update_plots() -> None:
    # Ensure img directories exist
    Path("img/2d").mkdir(parents=True, exist_ok=True)
    Path("img/3d").mkdir(parents=True, exist_ok=True)

    print("Generating 2D plots...")
    # 2D Plots logic from notebook
    for name, func in registry_original.items():
        print(f"Plotting 2D: {name}")
        fig, ax = plt.subplots(figsize=(6, 5))
        # Note: In the notebook, it seems they just call plot_2d directly.
        # We need to set the random key for reproducibility if needed,
        # but the plot_2d function likely handles execution.
        # Detailed inspection of notebook source shows:
        # plot_2d(func(ndim=2), title=name)
        # plt.savefig(f"img/2d/{name}.png", bbox_inches="tight")

        key = jr.key(0)
        plot_2d(func, key=key, ax=ax)
        ax.set_title(name)
        plt.savefig(f"img/2d/{name}.png", bbox_inches="tight")
        plt.close(fig)

    print("Generating 3D plots...")
    # 3D Plots logic from notebook
    for name, func in registry_original.items():
        print(f"Plotting 3D: {name}")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        key = jr.key(0)
        plot_3d(func, key=key, ax=ax)
        ax.set_title(name)
        plt.savefig(f"img/3d/{name}.png", bbox_inches="tight")
        plt.close(fig)

    # Generate 2D Overview Plot
    print("Generating 2D Overview Plot...")
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    axes = axes.flatten()

    for i, (name, func) in enumerate(registry_original.items()):
        ax = axes[i]
        key = jr.key(0)
        plot_2d(func, key=key, ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig("img/bbob_functions_overview_2d.png", dpi=150)
    plt.close(fig)

    # Generate 3D Overview Plot
    print("Generating 3D Overview Plot...")
    fig = plt.figure(figsize=(24, 30))

    for i, (name, func) in enumerate(registry_original.items()):
        ax = fig.add_subplot(6, 4, i + 1, projection="3d")
        key = jr.key(0)
        plot_3d(func, key=key, ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig("img/bbob_functions_overview_3d.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    update_plots()
