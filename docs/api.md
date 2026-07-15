# API Reference

## Problem Accessor

One lookup for everything a benchmark consumer needs: the callable, the
optimum location and value, the search-space bounds, the metadata tags and
the noise arity.

```python
import jax.random as jr
from bbob_jax import problem

p = problem("rastrigin", ndim=2, key=jr.key(0))
p.fn(p.x_opt)  # == p.f_opt
p.bounds       # (-5.0, 5.0)
p.tags         # {"separable": False, "unimodal": False}
p.noisy        # False — noisy functions are called as fn(x, key)
p.fn_true      # undisturbed value; is p.fn itself for noise-free functions
```

Pass `deterministic=True` for the zero-shift/identity-rotation instance
(the `*_original` registries construct the same instances).

For noisy functions (`p.noisy` is `True` — the BBOB-noisy suite and CEC
2005 F4/F17/F24/F25), `p.fn(x, key)` returns the disturbed value an
optimizer is allowed to see, while `p.fn_true(x)` returns the undisturbed
value (base + boundary penalty + `f_opt`) with the same bound instance
parameters — use it to measure true progress, COCO-style.

::: bbob_jax.problem

::: bbob_jax.Problem

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
- `bbob_jax.cec2005_function_characteristics`: Properties per function (unimodal/multimodal/composition/rotated flags, plus `noise` for the stochastic functions whose call signature is `fn(x, key)`, and `structure_modified` where the JAX implementation deviates from the official spec).

::: bbob_jax.cec2005_registry

::: bbob_jax.cec2005_registry_original

::: bbob_jax.cec2005_function_characteristics

## CEC 2005 Functions

Individual CEC 2005 benchmark function APIs (F1–F25). Access via the registries is recommended; the registry supplies internal parameters so the user-facing call is just `fn(x)`.

> **Note:** Parameters are generated from seeds rather than loaded from the official CEC 2005 data files. The stochastic functions (F4, F17, F24, F25) take a PRNG key as second argument: `fn(x, key)`. Functions F23–F25 replace the non-continuous rounding step with a smooth approximation (`structure_modified`). See `cec2005_function_characteristics` for per-function flags.

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

## BBOB-noisy Registry

Centralized access to the 30 BBOB-noisy benchmark functions (f101–f130)
and their metadata.

- `bbob_jax.bbob_noisy_registry`: Randomized variants of each BBOB-noisy function (names `bbob_noisy_f101` … `bbob_noisy_f130`). Every function is stochastic and called as `fn(x, key)`. Parameters are generated from seeds rather than derived from COCO instance IDs; the deterministic undisturbed path is cross-validated point-for-point against the compiled legacy reference code with reference-derived parameters injected (`scripts/crosscheck_bbob_noisy.py`).
- `bbob_jax.bbob_noisy_registry_original`: Deterministic *instance* variants (zero shift, identity rotations, zero `f_opt`) — the noise stays stochastic.
- `bbob_jax.bbob_noisy_function_characteristics`: Properties per function (`separable`/`unimodal` describe the undisturbed base; `gaussian_noise`/`uniform_noise`/`cauchy_noise` mark the noise model, exactly one per function; `severe` is False for the moderate group f101–f106; `noise` is always True).

The suite is eight base landscapes × three noise models: Gaussian
(multiplicative, log-normal), uniform (multiplicative) and Cauchy
(additive, heavy-tailed), applied to the residual above the optimum. The
×100 boundary penalty and `f_opt` are added outside the noise. Residuals
below `1e-8` bypass the noise entirely (the noise gate), so
`fn(x_opt, key) == f_opt` holds exactly. Semantics follow the legacy COCO
code (`benchmarksnoisy.c`), which matches the published definition; see
`docs/adr/0004`.

::: bbob_jax.bbob_noisy_registry

::: bbob_jax.bbob_noisy_registry_original

::: bbob_jax.bbob_noisy_function_characteristics

## BBOB-noisy Functions

Individual BBOB-noisy benchmark function APIs (f101–f130). Access via the
registries is recommended; the registry supplies internal parameters so the
user-facing call is just `fn(x, key)`. The `*_true` functions are the
undisturbed bases bound as `Problem.fn_true`.

::: bbob_jax._src.bbob_noisy.f101

::: bbob_jax._src.bbob_noisy.f102

::: bbob_jax._src.bbob_noisy.f103

::: bbob_jax._src.bbob_noisy.f104

::: bbob_jax._src.bbob_noisy.f105

::: bbob_jax._src.bbob_noisy.f106

::: bbob_jax._src.bbob_noisy.f107

::: bbob_jax._src.bbob_noisy.f108

::: bbob_jax._src.bbob_noisy.f109

::: bbob_jax._src.bbob_noisy.f110

::: bbob_jax._src.bbob_noisy.f111

::: bbob_jax._src.bbob_noisy.f112

::: bbob_jax._src.bbob_noisy.f113

::: bbob_jax._src.bbob_noisy.f114

::: bbob_jax._src.bbob_noisy.f115

::: bbob_jax._src.bbob_noisy.f116

::: bbob_jax._src.bbob_noisy.f117

::: bbob_jax._src.bbob_noisy.f118

::: bbob_jax._src.bbob_noisy.f119

::: bbob_jax._src.bbob_noisy.f120

::: bbob_jax._src.bbob_noisy.f121

::: bbob_jax._src.bbob_noisy.f122

::: bbob_jax._src.bbob_noisy.f123

::: bbob_jax._src.bbob_noisy.f124

::: bbob_jax._src.bbob_noisy.f125

::: bbob_jax._src.bbob_noisy.f126

::: bbob_jax._src.bbob_noisy.f127

::: bbob_jax._src.bbob_noisy.f128

::: bbob_jax._src.bbob_noisy.f129

::: bbob_jax._src.bbob_noisy.f130

::: bbob_jax._src.bbob_noisy.sphere_true

::: bbob_jax._src.bbob_noisy.rosenbrock_true

::: bbob_jax._src.bbob_noisy.step_ellipsoid_true

::: bbob_jax._src.bbob_noisy.ellipsoid_true

::: bbob_jax._src.bbob_noisy.different_powers_true

::: bbob_jax._src.bbob_noisy.schaffer_f7_true

::: bbob_jax._src.bbob_noisy.griewank_rosenbrock_true

::: bbob_jax._src.bbob_noisy.gallagher_true

## CEC 2017 Registry

Centralized access to the CEC 2017 benchmark functions and their metadata.

- `bbob_jax.cec2017_registry`: Randomized variants of each CEC 2017 function (names `cec2017_f1`, `cec2017_f3` … `cec2017_f30`; F2 was officially withdrawn). Parameters (shift vectors, rotation matrices, hybrid shuffle permutations) are generated from seeds rather than loaded from the official data files — results will not match published CEC 2017 benchmarking results, but the implementations are cross-validated point-for-point against the compiled official reference code with the official data injected (`scripts/crosscheck_cec2017.py`).
- `bbob_jax.cec2017_registry_original`: Deterministic baseline variants (zero shift, identity rotations, identity shuffles, no output offset).
- `bbob_jax.cec2017_function_characteristics`: Properties per function (`unimodal`/`multimodal`/`hybrid`/`composition`/`rotated`/`structure_modified` flags; the suite has no stochastic functions).

Some functions require a minimum dimensionality (`Problem.min_ndim`): the hybrids F11–F20 need one dimension per subcomponent kernel (up to 7), F29/F30 inherit their component hybrids' minimum, and F6 needs two. Makers raise `ValueError` below it.

Where the CEC 2017 technical report and the official reference code disagree, the implementation follows the code — that is what published results were produced with — and each divergence is documented in the function's docstring (e.g. F6's rotation is dead code in the reference, F8's non-continuity transform never executes, and F9's true minimizer is displaced from the sampled shift).

::: bbob_jax.cec2017_registry

::: bbob_jax.cec2017_registry_original

::: bbob_jax.cec2017_function_characteristics

## CEC 2017 Functions

Individual CEC 2017 benchmark function APIs (F1, F3–F30). Access via the registries is recommended; the registry supplies internal parameters so the user-facing call is just `fn(x)`.

::: bbob_jax._src.cec2017.f1

::: bbob_jax._src.cec2017.f3

::: bbob_jax._src.cec2017.f4

::: bbob_jax._src.cec2017.f5

::: bbob_jax._src.cec2017.f6

::: bbob_jax._src.cec2017.f7

::: bbob_jax._src.cec2017.f8

::: bbob_jax._src.cec2017.f9

::: bbob_jax._src.cec2017.f10

::: bbob_jax._src.cec2017.f11

::: bbob_jax._src.cec2017.f12

::: bbob_jax._src.cec2017.f13

::: bbob_jax._src.cec2017.f14

::: bbob_jax._src.cec2017.f15

::: bbob_jax._src.cec2017.f16

::: bbob_jax._src.cec2017.f17

::: bbob_jax._src.cec2017.f18

::: bbob_jax._src.cec2017.f19

::: bbob_jax._src.cec2017.f20

::: bbob_jax._src.cec2017.f21

::: bbob_jax._src.cec2017.f22

::: bbob_jax._src.cec2017.f23

::: bbob_jax._src.cec2017.f24

::: bbob_jax._src.cec2017.f25

::: bbob_jax._src.cec2017.f26

::: bbob_jax._src.cec2017.f27

::: bbob_jax._src.cec2017.f28

::: bbob_jax._src.cec2017.f29

::: bbob_jax._src.cec2017.f30

