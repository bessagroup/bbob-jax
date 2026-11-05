# Changelog

## 3.0.0 - 2026-11-05

### Changed (changes values)

- CEC 2005 F15–F25: the composition weights follow the official
  `calc_weight` (`src/def2.c`): every weight but the largest is damped by
  `1 - w_max^10`, then all are divided by their sum after damping. Before,
  the weights were rescaled so the largest was 1, which made the damping
  factor 0 (only the nearest component counted) and divided by the sum
  before damping. Values change away from the component optima;
  `f_opt`, the instances and task hashes do not. The weights are computed
  relative to the largest one in log space, so they never all underflow.
- CEC 2005 F15: the height normalization of each component is computed
  with identity matrices, as F15 evaluates and as the official code does.
  Before, it used the random rotations the factory binds but F15 ignores,
  which put values off by up to ~6e2 at D = 10. `f_opt`, the instances
  and task hashes do not change.
- CEC 2005 F17: the noise is `1 + 0.2*|N(0,1)|`, as in the official code
  (was `0.4`).
- CEC 2005 F24/F25: the noisy sphere component uses `1 + 0.1*|N(0,1)|`, as
  in the official code (was `0.01`), and its height normalization is fixed
  and noise-free. Before, it was recomputed on every call with the same
  draw, so the noise cancelled and `fn(x, key)` equalled `fn_true(x)`.
- `cec2005_deviations` no longer lists the composition weights or the
  F17/F24/F25 noise; F23–F25 keep their soft-rounding entries, and F24/F25
  gain one for their noise-free height normalization (the official code
  draws the noise once into it at setup).

### Added

- Point-for-point cross-checks against the official reference code, run in
  CI (`tests/test_crosscheck_*.py`): noiseless BBOB, BBOB-noisy, CEC 2017,
  CEC 2013 LSGO, and now CEC 2005 F15–F25 against the official C
  (C-Unix-linux-14-Mar-05). `scripts/fetch_references.sh DIR` downloads
  pinned, checksummed copies; the tests read `BBOB_JAX_REFERENCES`.
- `bbob_jax.cec2005_deviations`: per-function plain-language list of how
  the CEC 2005 suite departs from the official definitions.
