# Benchmark Functions for JAX

[View on GitHub](https://github.com/bessagroup/bbob-jax)

JAX implementations of five standard black-box optimization benchmark suites: **BBOB** noise-free and **BBOB-noisy** (Finck et al., 2009) [^1][^6], **CEC 2005** (Suganthan et al., 2005) [^4], **CEC 2017** (Awad et al., 2016) [^5] and **CEC 2013 LSGO** (Li et al., 2013) [^7] — 123 functions in total.

**First publication:** October 17, 2025

***

## Summary

This package provides JAX implementations of five widely-used benchmark suites for black-box optimization (see [Benchmark suites](#benchmark-suites) below). Every function is differentiable, JIT-compilable, and vectorizable via `vmap`, and is exposed through a simple registry that returns a ready-to-call objective together with its global minimum, as well as a `problem()` accessor that additionally bundles the optimum location, search-space bounds and metadata tags in one lookup. Both randomized (shifted and rotated) and deterministic factory variants are provided. Noisy functions are called as `fn(x, key)`; their undisturbed value is exposed as `Problem.fn_true` for COCO-style true-progress measurement.

## Statement of need

The BBOB and CEC benchmark suites are cornerstones of black-box optimization research, but their reference implementations are C and MATLAB codebases that cannot be differentiated, JIT-compiled or batched. This repository reimplements all five suites in JAX, enabling automatic differentiation, just-in-time (JIT) compilation, and XLA-accelerated performance — making them ideal for research in optimization, machine learning, and evolutionary algorithms. Where an official reference implementation exists, the port is cross-checked point-for-point against it (`scripts/crosscheck_*.py` in the repository).

## Benchmark suites

| Suite | Registry | Functions | Registry keys | Search space | Dimensions |
| --- | --- | --- | --- | --- | --- |
| **BBOB** noise-free [^1] | `registry`, `registry_original` | 24 | `sphere`, `rastrigin`, … `gallagher_101_peaks` | `[-5, 5]` | any `ndim` |
| **BBOB-noisy** [^6] | `bbob_noisy_registry`, `bbob_noisy_registry_original` | 30 | `bbob_noisy_f101` … `bbob_noisy_f130` | `[-5, 5]` | any `ndim` |
| **CEC 2005** [^4] | `cec2005_registry`, `cec2005_registry_original` | 25 | `f1` … `f25` | per function (`[-100, 100]`, `[-5, 5]`, `[-32, 32]`, `[-0.5, 0.5]`, `[-π, π]`, `[0, 600]`) | any `ndim` |
| **CEC 2017** [^5] | `cec2017_registry`, `cec2017_registry_original` | 29 | `cec2017_f1`, `cec2017_f3` … `cec2017_f30` | `[-100, 100]` | `min_ndim` 1–7, per function |
| **CEC 2013 LSGO** [^7] | `cec2013lsgo_registry` | 15 | `cec2013lsgo_f1` … `cec2013lsgo_f15` | per function (`[-100, 100]`, `[-5, 5]`, `[-32, 32]`) | fixed: 1000 (905 for F13/F14) |

Every suite also ships a per-function metadata dict and a bounds dict:
`function_characteristics` / `bbob_bounds`,
`bbob_noisy_function_characteristics` / `bbob_noisy_bounds`,
`cec2005_function_characteristics` / `cec2005_bounds`,
`cec2017_function_characteristics` / `cec2017_bounds`,
`cec2013lsgo_function_characteristics` / `cec2013lsgo_bounds`.

Suite-specific notes:

- **BBOB noise-free** — the 24 classic functions, tagged `separable` / `unimodal`.
- **BBOB-noisy** — f101–f130, all stochastic (`fn(x, key)`): Gaussian, uniform and Cauchy noise models at moderate (f101–f106) and severe severity. `Problem.fn_true` gives the undisturbed value. `*_original` fixes the *instance parameters*; the noise stays stochastic.
- **CEC 2005** — the 25 real-parameter functions: F1–F14 basic and expanded, F15–F25 hybrid compositions.
- **CEC 2017** — the 29 bound-constrained functions: 9 simple, 10 hybrid, 10 composition. F2 was officially withdrawn, and the numbering keeps the hole. Hybrid and composition functions need one dimension per subcomponent, so their `min_ndim` ranges from 2 to 7.
- **CEC 2013 LSGO** — 3 fully separable, 8 partially separable, 3 overlapping and 1 non-separable large-scale function. Unlike the other four, this is a **fixed-instance** suite: parameters are the official constants (vendored as package data, ported from [MetaBox](https://github.com/MetaEvo/MetaBox) — see [THIRD_PARTY_NOTICES.md](https://github.com/bessagroup/bbob-jax/blob/main/THIRD_PARTY_NOTICES.md)). Each function is defined only at its native dimension — `min_ndim` is the *only* valid `ndim`, not a floor — `key` is ignored, and there is no `_original` variant.

```python
import jax
import bbob_jax as bj

p = bj.problem("cec2017_f5", ndim=10, key=jax.random.key(0))
p.fn(p.x_opt), p.f_opt, p.bounds, p.tags, p.min_ndim
```

## Landscape galleries

Galleries are rendered for the three suites where a 2D view is meaningful.
BBOB-noisy is omitted (a single-sample plot of a stochastic function is not
informative — its undisturbed bases are the BBOB plots below), as is
CEC 2013 LSGO (defined only at 905 or 1000 dimensions).

<div align="center">
  <img src="img/bbob_functions_overview_3d.png" alt="BBOB functions 3D overview" width="80%">
  <br>
  <em>3D surface plots of the 24 BBOB benchmark functions.</em>
  <br><br>
  <img src="img/bbob_functions_overview_2d.png" alt="BBOB functions 2D overview" width="80%">
  <br>
  <em>2D contour plots of the 24 BBOB benchmark functions.</em>
  <br><br>
  <img src="img/cec2005_functions_overview_3d.png" alt="CEC 2005 functions 3D overview" width="80%">
  <br>
  <em>3D surface plots of the 25 CEC 2005 benchmark functions.</em>
  <br><br>
  <img src="img/cec2005_functions_overview_2d.png" alt="CEC 2005 functions 2D overview" width="80%">
  <br>
  <em>2D contour plots of the 25 CEC 2005 benchmark functions.</em>
  <br><br>
  <img src="img/cec2017_functions_overview_3d.png" alt="CEC 2017 functions 3D overview" width="80%">
  <br>
  <em>3D surface plots of the 29 CEC 2017 benchmark functions.</em>
  <br><br>
  <img src="img/cec2017_functions_overview_2d.png" alt="CEC 2017 functions 2D overview" width="80%">
  <br>
  <em>2D contour plots of the 29 CEC 2017 benchmark functions (F2 was
  officially withdrawn). Panels marked "(2D slice of nD)" belong to functions
  only defined from n dimensions up (one dimension per hybrid subcomponent
  kernel); they show a 2D slice of the smallest valid deterministic instance —
  the first two coordinates sweep the search range while the remaining
  coordinates stay pinned at the optimum plane.</em>
</div>


## Authorship

**Authors**:
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Authors affiliation:**
- Delft University of Technology (Bessa Research Group)

**Maintainer:**
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Maintainer affiliation:**
- Delft University of Technology (Bessa Research Group)


## Gradient-friendly implementations

Many BBOB functions use non-smooth operations (`abs`, `sign`, `sqrt`) that produce zero, undefined, or infinite gradients at certain points. This library uses [softjax](https://github.com/mvanderSchelling/softjax) straight-through estimators to provide well-defined gradients everywhere while keeping the forward pass *exactly* equal to the original function definitions.

For example, `jnp.abs(x)` has a zero gradient at `x = 0` and `jnp.sqrt(x)` has an infinite gradient at `x = 0`. The softjax replacements (`sj.abs_st`, `sj.sqrt`) return the exact same values but route gradients through smooth approximations during the backward pass. This means `jax.grad` produces useful, finite gradients without any loss of benchmark fidelity.

The following operations are replaced:

| Original | Replacement | Affected functions |
|---|---|---|
| `jnp.abs` | `sj.abs_st` | F2–F4, F10–F12, F14–F18, F21, F22 (via `tosz_func`, `tasy_func`) |
| `jnp.sign` | `sj.sign_st` | F2–F4, F10, F11, F15, F16, F21, F22 (via `tosz_func`) |
| `jnp.sqrt` | `sj.sqrt` | F3, F12, F13, F15, F17, F18, F20 |
| `jnp.maximum(., 0)` | `sj.relu_st` | F4, F7, F16–F18, F20–F24 (via `penalty`) |
| `jnp.where` / `>` | `sj.where` / `sj.greater_st` | F3, F6, F12, F15, F17, F18 |

softjax is used only where JAX's own gradient is degenerate (zero, undefined, or infinite). Operations with well-defined subgradients — in particular the `max`/`min` reductions in F21, F22, F24 — use plain `jnp.max`/`jnp.min`: their gradient flows through the selected element, which is the meaningful descent direction, and the straight-through soft-sort would otherwise run in every forward pass (the soft branch of `stop_gradient(hard - soft) + soft` cannot be dead-code-eliminated).

Functions that are *intentionally* non-smooth (F7 `step_ellipsoid`, F23 `katsuura`) are left unchanged — smoothing them would defeat their benchmarking purpose.

## Getting started

To install the package, use pip:

```bash
pip install bbob-jax
```

After installation, see the full guide: [Getting Started](usage.ipynb).

## Related Work

This project builds on and complements established benchmarking efforts and tooling in black-box optimization. The resources below are closely related and provide broader context and utilities.

- [COCO platform (COmparing Continuous Optimisers)](https://coco-platform.org/): benchmarking framework and tools for black-box optimization. [^2]
- [EvoSax](https://github.com/RobertTLange/evosax): JAX-based evolution strategies library that includes BBOB function support and benchmarking utilities. [^3]

## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/bessagroup/bbob-jax/issues) to report them.

## Citation

--8<-- ".citation.md"

## License

Copyright (c) 2025, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/bbob-jax/blob/main/LICENSE) for the full license text.

`bbob-jax` also incorporates third-party components under their own terms; see [THIRD_PARTY_NOTICES.md](https://github.com/bessagroup/bbob-jax/blob/main/THIRD_PARTY_NOTICES.md).

[^1]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions](https://inria.hal.science/inria-00362633v2/document), INRIA.

[^2]: Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021), COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting. Optimization Methods and Software, 36(1), 114–144. https://doi.org/10.1080/10556788.2020.1808977

[^3]: Lange, R. T. (2022), evosax: JAX-based Evolution Strategies. arXiv preprint [arXiv:2212.04180](https://arxiv.org/abs/2212.04180).

[^4]: Suganthan, P. N., Hansen, N., Liang, J. J., and Deb, K. (2005), Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization.

[^5]: Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., and Suganthan, P. N. (2016), Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Real-Parameter Numerical Optimization.

[^6]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noisy functions definitions](https://inria.hal.science/inria-00369466v2/document), INRIA.

[^7]: Li, X., Tang, K., Omidvar, M. N., Yang, Z., and Qin, K. (2013), Benchmark Functions for the CEC 2013 Special Session and Competition on Large-Scale Global Optimization. Technical Report, RMIT University.

