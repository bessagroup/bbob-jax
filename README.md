# BBOB Benchmark set for JAX

| [**GitHub**](https://github.com/bessagroup/bbob-jax)
| [**PyPI**](https://pypi.org/project/bbob-jax/)
| [**Documentation**](https://bbob-jax.readthedocs.io/)
| [**Zenodo**](https://doi.org/10.5281/zenodo.17426894) 

JAX implementation of the BBOB Benchmark functions for black-box optimization, based on the original definitions by Finck et al. (2009) [^1].

**First publication:** October 17, 2025

***

## Statement of need

This repository provides the original BBOB 24 noise-free, real-parameter, single-objective benchmark functions reimplemented in JAX. Originally written in C, these functions have been translated to JAX to enable automatic differentiation, just-in-time (JIT) compilation, and XLA-accelerated performance; making them ideal for research in optimization, machine learning, and evolutionary algorithms.

<div align="center">
  <img src="img/bbob_functions_overview_3d.png" alt="BBOB functions 3D overview" width="80%">
  <br>
  <em>3D surface plots of the 24 BBOB benchmark functions.</em>
  <br><br>
  <img src="img/bbob_functions_overview_2d.png" alt="BBOB functions 2D overview" width="80%">
  <br>
  <em>2D contour plots of the 24 BBOB benchmark functions.</em>
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

**Zenodo** ([link](https://doi.org/10.5281/zenodo.17426894)):
```bibtex
@software{vanderSchelling2025,
  title        = {Black-box optimization benchmarking (bbob) problem
                   set for JAX},
  author       = {van der Schelling, M. P. and Bessa, M A.},
  month        = {nov},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.17426894},
  url          = {https://doi.org/10.5281/zenodo.17426894},
}
```

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

## License

Copyright (c) 2025, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/bbob-jax/blob/main/LICENSE) for the full license text.

[^1]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions](https://inria.hal.science/inria-00362633v2/document), INRIA.

[^2]: Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., and Brockhoff, D. (2021), COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting. Optimization Methods and Software, 36(1), 114–144. https://doi.org/10.1080/10556788.2020.1808977

[^3]: Lange, R. T. (2022), evosax: JAX-based Evolution Strategies. arXiv preprint [arXiv:2212.04180](https://arxiv.org/abs/2212.04180).