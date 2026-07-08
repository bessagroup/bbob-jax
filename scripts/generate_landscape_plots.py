"""Generate randomized 3D BBOB loss-landscape PNGs grouped by COCO category.

Each PNG is the 3D loss landscape of a BBOB function in 2D input space, using
the randomized ``registry`` (random x_opt / f_opt / rotations), noise-free,
with no title, axes, or background -- just the surface. Output is
high-resolution (DPI 300) transparent PNG suitable for slides.

Layout::

    img/landscapes/<category>/<function_name>.png

Run with::

    uv run --no-sync python scripts/generate_landscape_plots.py
"""

from pathlib import Path

import jax.random as jr
import matplotlib.pyplot as plt

from bbob_jax import bbob_bounds, registry
from bbob_jax.plotting import plot_3d

# Mesh resolution (points per axis) for the surface. Higher = finer grid.
# plot_3d defaults to 300; we use a denser mesh for smoother,
# slide-quality plots.
PX = 600

# Canonical COCO/BBOB grouping of the 24 noise-free
# functions into 5 categories.
# (tags.py only stores separable/unimodal booleans, so the 5-group split is
# encoded explicitly here.)
CATEGORIES: dict[str, list[str]] = {
    "1_separable": [
        "sphere",
        "ellipsoid_seperable",
        "rastrigin_seperable",
        "skew_rastrigin_bueche",
        "linear_slope",
    ],
    "2_low_moderate_conditioning": [
        "attractive_sector",
        "step_ellipsoid",
        "rosenbrock",
        "rosenbrock_rotated",
    ],
    "3_high_conditioning_unimodal": [
        "ellipsoid",
        "discuss",
        "bent_cigar",
        "sharp_ridge",
        "sum_of_different_powers",
    ],
    "4_multimodal_adequate_structure": [
        "rastrigin",
        "weierstrass",
        "schaffer_f7_condition_10",
        "schaffer_f7_condition_1000",
        "griewank_rosenbrock_f8f2",
    ],
    "5_multimodal_weak_structure": [
        "schwefel_xsinx",
        "gallagher_101_peaks",
        "gallagher_21_peaks",
        "katsuura",
        "lunacek_bi_rastrigin",
    ],
}


def generate_landscape_plots() -> None:
    # Fail loudly if the category map drifts from the registry.
    mapped = {name for names in CATEGORIES.values() for name in names}
    missing = set(registry) - mapped
    unknown = mapped - set(registry)
    if missing or unknown:
        raise ValueError(
            f"Category map out of sync with registry: "
            f"missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    for category, names in CATEGORIES.items():
        out_dir = Path("img/landscapes") / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in names:
            print(f"Plotting 3D landscape: {category}/{name}")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Randomized instance (random x_opt / f_opt / rotations),
            # 2D, noise-free.
            plot_3d(
                registry[name],
                key=jr.key(0),
                bounds=bbob_bounds[name],
                px=PX,
                ax=ax,
            )

            # Strip everything but the surface: panes, spines, grid,
            # ticks, labels.
            ax.set_axis_off()

            fig.savefig(
                out_dir / f"{name}.png",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)


if __name__ == "__main__":
    generate_landscape_plots()
