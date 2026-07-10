# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bbob-jax` is a JAX implementation of the 24 BBOB (Black-Box Optimization Benchmark) noise-free functions, the 25 CEC 2005 benchmark functions, and the CEC 2017 bound-constrained suite (in progress: F1, F3–F20 landed; compositions F21–F30 follow; F2 was officially withdrawn and is skipped). The key value-add over the original C implementations is full JAX compatibility: automatic differentiation, JIT compilation, and vectorization via `vmap`.

## Commands

```bash
uv sync                    # Install all dependencies
uv run --no-sync pytest    # Run full test suite (or: make test)
uv run --no-sync pytest tests/test_example.py::test_name  # Run single test
make lint                  # Run Ruff linter
make format                # Format code with Ruff
make docs                  # Start MkDocs dev server
```

## Architecture

### Public API

`src/bbob_jax/__init__.py` exports:
- 24 named BBOB functions (e.g. `sphere`, `rastrigin`, `rosenbrock`)
- `problem` / `Problem` — one-lookup accessor bundling fn, x_opt, f_opt, bounds, tags, noise arity
- `registry` — randomized BBOB factory (random x_opt, f_opt, rotation matrices)
- `registry_original` — deterministic BBOB factory (zero x_opt, zero f_opt, identity rotations)
- `cec2005_registry` — randomized CEC 2005 factory (25 functions `f1`–`f25`)
- `cec2005_registry_original` — deterministic CEC 2005 factory
- `cec2017_registry` / `cec2017_registry_original` — CEC 2017 factories (names `cec2017_f1`, `cec2017_f3`…; prefixed because `SPEC_BY_NAME` is a single namespace and CEC 2005 owns the bare `fN` keys)
- `function_characteristics` — BBOB metadata dict (separable/unimodal flags per function)
- `cec2005_function_characteristics` — CEC 2005 metadata dict
- `cec2017_function_characteristics` — CEC 2017 metadata dict (`unimodal`/`multimodal`/`hybrid`/`composition`/`rotated`/`structure_modified`; no `noise` key)
- `bbob_bounds` / `cec2005_bounds` / `cec2017_bounds` — per-function search-space bounds (also via the `bounds` submodule)

### Spec Table (single source of truth)

`_src/spec.py` holds one `FunctionSpec` row per function: implementation,
factory (`maker`), tags, bounds, and an `x_opt_from` resolver locating the
true optimum (e.g. `linear_slope`'s optimum is `_ls_x_opt`, compositions use
the first component's optimum). The registries, tag dicts and bounds dicts
are derived views of this table — **adding a function means implementing it
in `bbob.py`/`cec2005.py` and adding one spec row**; tests parametrize off
the registries and pick it up automatically. See `docs/adr/0002`.

### Registry Pattern

The registries map `function_name → factory`, where each factory is called as:
```python
fn, f_opt = registry["sphere"](ndim=2, key=jax_key)
# fn: Callable[[jax.Array], jax.Array] — partially applied with x_opt, f_opt, R, Q
# f_opt: jax.Array — the global minimum value
```

Factories live in `_src/factories.py` and use `jax.tree_util.Partial` to
bind parameters. Every factory takes a `deterministic` flag; the
`*_original` registries are the same makers with `deterministic=True`
bound (zero shift, identity rotations, zero f_opt).

### Problem Accessor

```python
p = bbob_jax.problem("rastrigin", ndim=2, key=jax_key)  # deterministic=True for the *_original instance
p.fn, p.x_opt, p.f_opt, p.bounds, p.tags, p.noisy
```
`fn(p.x_opt) == p.f_opt` holds for every function (documented exceptions:
deterministic CEC compositions are degenerate). Noisy CEC functions
(`noise` tag) are called as `fn(x, key)`. `Problem.min_ndim` is the
smallest supported dimension (default 1); makers raise `ValueError` below
it (e.g. `cec2017_f6` needs `ndim >= 2`; CEC 2017 hybrids need one
dimension per subcomponent kernel and up to 7 where the chunk split
demands it — see `cec2017_hybrid_partition`).

### Core Function Signature

All 24 BBOB functions in `_src/bbob.py`, the 25 CEC 2005 functions in `_src/cec2005.py` and the CEC 2017 functions in `_src/cec2017.py` share the internal signature:
```python
def fn(x, x_opt, f_opt, R, Q) -> jax.Array
```
After partial application via the registry, the user-facing signature is just `fn(x)`. (Some factories also bind extra precomputed keyword arguments, e.g. `_mat` or `_f_max`.)

BBOB functions compose transformations from `_src/transforms.py`:
- Shift by `x_opt`, rotate with `R`/`Q`
- Apply `tosz_func` (smooth log/sine deformation) and `tasy_func` (asymmetry)
- Apply `lambda_func` (diagonal conditioning)
- Add boundary `penalty` and offset by `f_opt`

This internal signature is deliberately kept as-is (including its warts —
see `docs/adr/0001`): problem instances are pinned to keys by downstream
databanks, so the parameter derivation must not change.

Gradient convention: softjax (`sj.*`) straight-through estimators are used only where JAX's own gradient is degenerate (zero, undefined, or infinite — e.g. `sign`, `round`, `abs`, `sqrt` at 0). Operations with well-defined subgradients, in particular `max`/`min` reductions, use plain `jnp` — the straight-through soft branch would otherwise execute in every forward pass and inflate grad-compile time (see the Gallagher functions).

### Module Layout

```
src/bbob_jax/
├── __init__.py          # Public exports
├── plotting.py          # Public plotting API (wraps _src/plotting.py)
├── bounds.py            # Public bounds API (re-exports bbob_bounds, cec2005_bounds)
└── _src/
    ├── bbob.py          # All 24 BBOB function implementations
    ├── cec2005.py       # All 25 CEC 2005 function implementations (f1–f25)
    ├── cec2017.py       # CEC 2017 function implementations (F2 withdrawn/skipped)
    ├── spec.py          # FunctionSpec table — single source of truth per function
    ├── factories.py     # Mode-parameterized makers (deterministic= flag)
    ├── registry.py      # The six registry dicts, derived from spec.py
    ├── problem.py       # Problem record + problem() accessor
    ├── transforms.py    # BBOB transforms: tosz, tasy, lambda, penalty
    ├── composition.py   # CEC kernels (2005 + 2017) + hybrid composition machinery
    ├── sampling.py      # fopt, xopt, bernoulli_vector, rotation_matrix
    ├── mesh.py          # _create_mesh grid evaluator (matplotlib-free)
    ├── tags.py          # function_characteristics, derived from spec.py
    ├── cec2005_tags.py  # cec2005_function_characteristics, derived from spec.py
    ├── cec2017_tags.py  # cec2017_function_characteristics, derived from spec.py
    ├── bounds.py        # bbob_bounds, cec2005_bounds, cec2017_bounds, derived from spec.py
    └── plotting.py      # plot_2d, plot_3d (requires optional matplotlib dep)
```

Architecture decisions are recorded in `docs/adr/`.

### Testing

Tests live in `tests/` (`test_example.py` for BBOB, `test_cec2005.py` for CEC 2005, `test_cec2017.py` for CEC 2017, plus `test_bounds.py`, `test_composition.py`, `test_problem.py`, `test_consistency.py`, `test_gallagher_equivalence.py`). They are parametrized over all functions × multiple dimensions × both registries. Each test validates:
- Correct scalar output shape
- JIT compilation (`jax.jit`)
- Vectorization (`jax.vmap`)
- Gradient computation (`jax.grad`)
- NaN propagation (NaN in → NaN out) for BBOB and CEC 2017
- `fn(x_opt) == f_opt` for all suites via `problem()` (`test_problem.py`)
- Registry/tags/bounds key-set consistency and tag schemas (`test_consistency.py`)

`matplotlib` is an optional dependency (install group `plot`); tests don't require it.

## Related repositories

`bbob-jax` provides the benchmark functions used across the L2CO ecosystem (Bessa Research Group). Related repositories:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
