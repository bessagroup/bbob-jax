# API Reference

## Function Registry

Centralized access to the benchmark functions and their metadata.

- `bbob_jax.registry`: Randomized variants of each function. Call with `x` and `key` to get a reproducible stochastic instance (random shifts/rotations and fopt).
- `bbob_jax.registry_original`: Deterministic baseline variants (no random shift/rotation, no output offset). Useful for debugging and reference.
- `bbob_jax.function_characteristics`: Loss-landscape properties per function (e.g., separability, conditioning, modality) to filter or group benchmarks.

::: bbob_jax.registry

::: bbob_jax.registry_original

::: bbob_jax.function_characteristics

## Plotting Utilities

Helpers to quickly visualize functions in 2D and 3D. These utilities evaluate a provided benchmark function over a grid and render either a heatmap or a surface plot.

- `plot_2d`: Renders a log-normalized heatmap of the function landscape.
- `plot_3d`: Renders a 3D surface; z-values are sym-log normalized for readability.

::: bbob_jax.plotting.plot_2d

::: bbob_jax.plotting.plot_3d

## BBOB Functions

Individual benchmark function APIs. Public call pattern is via the root package (e.g., `bbob_jax.sphere`). When used through the registries, call as `fn(x, key=...)`; the registry supplies internal shift/rotation parameters so you only provide the decision vector `x` (shape `(..., dim)`) and an optional PRNG `key`.

::: bbob_jax.sphere

::: bbob_jax.ellipsoid_seperable

::: bbob_jax.rastrigin_seperable

::: bbob_jax.skew_rastrigin_bueche

::: bbob_jax.linear_slope

::: bbob_jax.attractive_sector

::: bbob_jax.step_ellipsoid

::: bbob_jax.rosenbrock

::: bbob_jax.rosenbrock_rotated

::: bbob_jax.ellipsoid

::: bbob_jax.discuss

::: bbob_jax.bent_cigar

::: bbob_jax.sharp_ridge

::: bbob_jax.sum_of_different_powers

::: bbob_jax.rastrigin

::: bbob_jax.weierstrass

::: bbob_jax.schaffer_f7_condition_10

::: bbob_jax.schaffer_f7_condition_1000

::: bbob_jax.griewank_rosenbrock_f8f2

::: bbob_jax.schwefel_xsinx

::: bbob_jax.gallagher_101_peaks

::: bbob_jax.gallagher_21_peaks

::: bbob_jax.katsuura

::: bbob_jax.lunacek_bi_rastrigin