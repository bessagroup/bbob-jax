# Benchmark Functions for JAX

[View on GitHub](https://github.com/bessagroup/bbob-jax)

JAX implementations of the **BBOB** benchmark functions (Finck et al., 2009) [^1] and the **CEC 2005** benchmark functions (Suganthan et al., 2005) [^4] for black-box optimization.

**First publication:** October 17, 2025

***

## Summary

This package provides JAX implementations of two widely-used benchmark suites for black-box optimization: the **BBOB** 24 noise-free functions (Finck et al., 2009) [^1] and the **CEC 2005** 25 real-parameter functions (Suganthan et al., 2005) [^4]. All functions support automatic differentiation, JIT compilation, and XLA-accelerated evaluation.

## Statement of need

The BBOB and CEC 2005 benchmark suites are cornerstones of black-box optimization research. This repository provides JAX reimplementations of both: the 24 BBOB noise-free functions originally written in C, and the 25 CEC 2005 real-parameter functions. Translating these suites to JAX enables automatic differentiation, just-in-time (JIT) compilation, and XLA-accelerated performance — making them ideal for research in optimization, machine learning, and evolutionary algorithms.

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

[^1]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions](https://inria.hal.science/inria-00362633v2/document), INRIA.

[^2]: Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021), COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting. Optimization Methods and Software, 36(1), 114–144. https://doi.org/10.1080/10556788.2020.1808977

[^3]: Lange, R. T. (2022), evosax: JAX-based Evolution Strategies. arXiv preprint [arXiv:2212.04180](https://arxiv.org/abs/2212.04180).

[^4]: Suganthan, P. N., Hansen, N., Liang, J. J., and Deb, K. (2005), Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization.

