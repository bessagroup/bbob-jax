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

## CEC 2005 Registry

Centralized access to the CEC 2005 benchmark functions and their metadata.

- `bbob_jax.cec2005_registry`: Randomized variants of each CEC 2005 function. Parameters (shift vectors, rotation matrices) are generated from seeds rather than loaded from the official CEC 2005 data files — results will not match published CEC 2005 benchmarking results.
- `bbob_jax.cec2005_registry_original`: Deterministic baseline variants (no random shift/rotation, no output offset). Useful for debugging and reference.
- `bbob_jax.cec2005_function_characteristics`: Properties per function (unimodal/multimodal/composition/rotated flags, plus `noise_omitted` and `structure_modified` flags where the JAX implementation deviates from the official spec).

::: bbob_jax.cec2005_registry

::: bbob_jax.cec2005_registry_original

::: bbob_jax.cec2005_function_characteristics

## CEC 2005 Functions

Individual CEC 2005 benchmark function APIs (F1–F25). Access via the registries is recommended; the registry supplies internal parameters so the user-facing call is just `fn(x)`.

> **Note:** Parameters are generated from seeds rather than loaded from the official CEC 2005 data files. Functions F4 and F17 have Gaussian noise omitted for `jax.grad` compatibility. Function F23 has its rounding step removed. See `cec2005_function_characteristics` for per-function flags.

::: bbob_jax._src.cec2005.f1

::: bbob_jax._src.cec2005.f2

::: bbob_jax._src.cec2005.f3

::: bbob_jax._src.cec2005.f4

::: bbob_jax._src.cec2005.f5

::: bbob_jax._src.cec2005.f6

::: bbob_jax._src.cec2005.f7

::: bbob_jax._src.cec2005.f8

::: bbob_jax._src.cec2005.f9

::: bbob_jax._src.cec2005.f10

::: bbob_jax._src.cec2005.f11

::: bbob_jax._src.cec2005.f12

::: bbob_jax._src.cec2005.f13

::: bbob_jax._src.cec2005.f14

::: bbob_jax._src.cec2005.f15

::: bbob_jax._src.cec2005.f16

::: bbob_jax._src.cec2005.f17

::: bbob_jax._src.cec2005.f18

::: bbob_jax._src.cec2005.f19

::: bbob_jax._src.cec2005.f20

::: bbob_jax._src.cec2005.f21

::: bbob_jax._src.cec2005.f22

::: bbob_jax._src.cec2005.f23

::: bbob_jax._src.cec2005.f24

::: bbob_jax._src.cec2005.f25

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