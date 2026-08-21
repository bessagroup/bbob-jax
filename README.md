# Benchmark Functions for JAX

| [**GitHub**](https://github.com/bessagroup/bbob-jax)
| [**PyPI**](https://pypi.org/project/bbob-jax/)
| [**Documentation**](https://bbob-jax.readthedocs.io/)
| [**Zenodo**](https://doi.org/10.5281/zenodo.17426893) 

JAX implementations of five standard black-box optimization benchmark suites: **BBOB** noise-free and **BBOB-noisy** (Finck et al., 2009) [^1][^6], **CEC 2005** (Suganthan et al., 2005) [^4], **CEC 2017** (Awad et al., 2016) [^5] and **CEC 2013 LSGO** (Li et al., 2013) [^7] — 123 functions in total.

**First publication:** October 17, 2025

***

## Summary

`bbob-jax` is a pure-[JAX](https://github.com/jax-ml/jax) reimplementation of five standard black-box optimization benchmark suites (see [Benchmark suites](#benchmark-suites) below). Every function is differentiable, JIT-compilable, and vectorizable via `vmap`, and is exposed through a simple registry that returns a ready-to-call objective together with its global minimum, as well as a `problem()` accessor that additionally bundles the optimum location, search-space bounds and metadata tags in one lookup. Both randomized (shifted and rotated) and deterministic factory variants are provided. Noisy functions are called as `fn(x, key)`; their undisturbed value is exposed as `Problem.fn_true` for COCO-style true-progress measurement.

## Statement of need

The BBOB and CEC benchmark suites are cornerstones of black-box optimization research, but their reference implementations are C and MATLAB codebases that cannot be differentiated, JIT-compiled or batched. This repository reimplements all five suites in JAX, enabling automatic differentiation, just-in-time (JIT) compilation, and XLA-accelerated performance — making them ideal for research in optimization, machine learning, and evolutionary algorithms. Where an official reference implementation exists, the port is cross-checked point-for-point against it (`scripts/crosscheck_*.py`).

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
- **CEC 2013 LSGO** — 3 fully separable, 8 partially separable, 3 overlapping and 1 non-separable large-scale function. Unlike the other four, this is a **fixed-instance** suite: parameters are the official constants (vendored as package data), each function is defined only at its native dimension, `key` is ignored, and there is no `_original` variant. See [Acknowledgements](#acknowledgements--third-party-components).

```python
import jax
import bbob_jax as bj

p = bj.problem("cec2017_f5", ndim=10, key=jax.random.key(0))
p.fn(p.x_opt), p.f_opt, p.bounds, p.tags, p.min_ndim
```

<div align="center">
  <img src="img/bbob_functions_overview_3d.png" alt="BBOB functions 3D overview" width="80%">
  <br>
  <em>3D surface plots of the 24 BBOB benchmark functions. The full landscape
  galleries (BBOB, CEC 2005 and CEC 2017, in 2D and 3D) are in the
  <a href="https://bbob-jax.readthedocs.io/">documentation</a>.</em>
</div>

## Authorship & Citation

**Authors**:
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Authors affiliation:**
- Delft University of Technology (Bessa Research Group)

**Maintainer:**
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Maintainer affiliation:**
- Delft University of Technology (Bessa Research Group)

If you use `bbob-jax` in your research or in a scientific publication, it is appreciated that you cite the paper below:

**Zenodo** ([link](https://doi.org/10.5281/zenodo.17426893)):
```bibtex
@software{vanderSchelling2025,
  title        = {Black-box optimization benchmarking (bbob) problem
                   set for JAX},
  author       = {van der Schelling, M. P. and Bessa, M A.},
  month        = {jul},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.17426893},
  url          = {https://doi.org/10.5281/zenodo.17426893},
}
```

## Gradient-friendly implementations

Many BBOB functions use non-smooth operations (`abs`, `sign`, `sqrt`, `max`, `min`) that produce zero, undefined, or infinite gradients at certain points. This library uses [softjax](https://github.com/mvanderSchelling/softjax) straight-through estimators to provide well-defined gradients everywhere while keeping the forward pass *exactly* equal to the original function definitions.

For example, `jnp.abs(x)` has a zero gradient at `x = 0` and `jnp.sqrt(x)` has an infinite gradient at `x = 0`. The softjax replacements (`sj.abs_st`, `sj.sqrt`) return the exact same values but route gradients through smooth approximations during the backward pass. This means `jax.grad` produces useful, finite gradients without any loss of benchmark fidelity.

Functions that are *intentionally* non-smooth (F7 `step_ellipsoid`, F23 `katsuura`) are left unchanged — smoothing them would defeat their benchmarking purpose.

## Getting started

To install the package, use pip:

```bash
pip install bbob-jax
```

## Related Work

This project builds on and complements established benchmarking efforts and tooling in black-box optimization. The resources below are closely related and provide broader context and utilities.

- [COCO platform (COmparing Continuous Optimisers)](https://coco-platform.org/): benchmarking framework and tools for black-box optimization. [^2]
- [EvoSax](https://github.com/RobertTLange/evosax): JAX-based evolution strategies library that includes BBOB function support and benchmarking utilities. [^3]

## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/bessagroup/bbob-jax/issues) to report them.

## Acknowledgements & third-party components

The **CEC 2013 Large-Scale Global Optimization** suite (`cec2013lsgo_registry`,
functions F1–F15) was ported to JAX from [MetaBox](https://github.com/MetaEvo/MetaBox)'s
NumPy implementation (`MetaEvo/MetaBox@5565a28`, BSD 3-Clause, © 2023 MetaEvolution Lab),
which in turn derives from Daniel Molina's [`cec2013lsgo`](https://github.com/dmolina/cec2013lsgo)
reference code and the official CEC 2013 competition data. The vendored
benchmark constants and full provenance are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
`src/bbob_jax/_src/cec2013lsgo_data/PROVENANCE.md`.

If you use the LSGO suite, please cite the original benchmark, MetaBox, and
`bbob-jax`:

```bibtex
@techreport{Li2013LSGO,
  title       = {Benchmark Functions for the CEC 2013 Special Session and
                 Competition on Large-Scale Global Optimization},
  author      = {Li, Xiaodong and Tang, Ke and Omidvar, Mohammad Nabi and
                 Yang, Zhenyu and Qin, Kai},
  institution = {RMIT University},
  year        = {2013},
}

@inproceedings{Ma2023MetaBox,
  title     = {MetaBox: A Benchmark Platform for Meta-Black-Box Optimization
               with Reinforcement Learning},
  author    = {Ma, Zeyuan and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  note      = {arXiv:2311.02708},
}
```

MetaBox-v2 (NeurIPS 2025, arXiv:2505.17745) extends the platform used above.

## License

Copyright (c) 2025, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/bbob-jax/blob/main/LICENSE) for the full license text.

`bbob-jax` also incorporates third-party components under their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

[^1]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions](https://inria.hal.science/inria-00362633v2/document), INRIA.

[^2]: Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021), COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting. Optimization Methods and Software, 36(1), 114–144. https://doi.org/10.1080/10556788.2020.1808977

[^3]: Lange, R. T. (2022), evosax: JAX-based Evolution Strategies. arXiv preprint [arXiv:2212.04180](https://arxiv.org/abs/2212.04180).

[^4]: Suganthan, P. N., Hansen, N., Liang, J. J., and Deb, K. (2005), Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization.

[^5]: Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., and Suganthan, P. N. (2016), Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Real-Parameter Numerical Optimization.

[^6]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noisy functions definitions](https://inria.hal.science/inria-00369466v2/document), INRIA.

[^7]: Li, X., Tang, K., Omidvar, M. N., Yang, Z., and Qin, K. (2013), Benchmark Functions for the CEC 2013 Special Session and Competition on Large-Scale Global Optimization. Technical Report, RMIT University.

## Related repositories

`bbob-jax` provides the benchmark functions used across the L2CO ecosystem developed in the [Bessa Research Group](https://github.com/bessagroup). The repositories below work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB, BBOB-noisy, CEC 2005, CEC 2017 and CEC 2013 LSGO black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.