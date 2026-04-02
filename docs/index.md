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

