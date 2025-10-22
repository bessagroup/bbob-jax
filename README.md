# BBOB Benchmark set for Jax

| [**GitHub**](https://github.com/mpvanderschelling/bbob-jax)
| [**PyPi**](https://pypi.org/project/bbob-jax/)
|

JAX implementation of the BBOB Benchmark functions for black-box optimization, based on the original definitions by Finck et al. (2009) [^1].

[^1]: Finck, S., Hansen, N., Ros, R., and Auger, A. (2009), [Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions](https://inria.hal.science/inria-00362633v2/document), INRIA. 

**First publication:** October 17, 2025

***

## Summary

This repository providess the original BBOB 24 noise-free, real-parameter, single-objective benchmark functions reimplemented in JAX. Originally written in C, these functions have been translated to JAX to enable automatic differentiation, just-in-time (JIT) compilation, and XLA-accelerated performance — making them ideal for research in optimization, machine learning, and evolutionary algorithms.

## Statement of need

<!-- Write here the statement of need for this package -->

## Authorship

**Authors**:
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Authors afilliation:**
- Delft University of Technology (Bessa Research Group)

**Maintainer:**
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Maintainer afilliation:**
- Delft University of Technology (Bessa Research Group)


## Getting started

To install the package, use pip:

```bash
pip install bbob-jax
```

## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/mpvanderschelling/bbob_jax/issues) to report them.

## License

Copyright (c) 2025, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/mpvanderschelling/bbob_jax/blob/main/LICENSE) for the full license text.

