# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bbob-jax` is a JAX implementation of the 24 BBOB (Black-Box Optimization Benchmark) noise-free functions and the 25 CEC 2005 benchmark functions. The key value-add over the original C implementations is full JAX compatibility: automatic differentiation, JIT compilation, and vectorization via `vmap`.

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
- `registry` — randomized BBOB factory (random x_opt, f_opt, rotation matrices)
- `registry_original` — deterministic BBOB factory (zero x_opt, zero f_opt, identity rotations)
- `cec2005_registry` — randomized CEC 2005 factory (25 functions `f1`–`f25`)
- `cec2005_registry_original` — deterministic CEC 2005 factory
- `function_characteristics` — BBOB metadata dict (separable/unimodal flags per function)
- `cec2005_function_characteristics` — CEC 2005 metadata dict
- `bbob_bounds` / `cec2005_bounds` — per-function search-space bounds (also via the `bounds` submodule)

### Registry Pattern

Both registries map `function_name → factory`, where each factory is called as:
```python
fn, f_opt = registry["sphere"](ndim=2, key=jax_key)
# fn: Callable[[jax.Array], jax.Array] — partially applied with x_opt, f_opt, R, Q
# f_opt: jax.Array — the global minimum value
```

Implemented in `_src/registry.py` using `jax.tree_util.Partial` to bind parameters.

### Core Function Signature

All 24 BBOB functions in `_src/bbob.py` and the 25 CEC 2005 functions in `_src/cec2005.py` share the internal signature:
```python
def fn(x, x_opt, f_opt, R, Q) -> jax.Array
```
After partial application via the registry, the user-facing signature is just `fn(x)`. (Some factories also bind extra precomputed keyword arguments, e.g. `_mat` or `_f_max`.)

Functions compose transformations from `_src/utils.py`:
- Shift by `x_opt`, rotate with `R`/`Q`
- Apply `tosz_func` (smooth log/sine deformation) and `tasy_func` (asymmetry)
- Apply `lambda_func` (diagonal conditioning)
- Add boundary `penalty` and offset by `f_opt`

### Module Layout

```
src/bbob_jax/
├── __init__.py          # Public exports
├── plotting.py          # Public plotting API (wraps _src/plotting.py)
├── bounds.py            # Public bounds API (re-exports bbob_bounds, cec2005_bounds)
└── _src/
    ├── bbob.py          # All 24 BBOB function implementations
    ├── cec2005.py       # All 25 CEC 2005 function implementations (f1–f25)
    ├── registry.py      # BBOB + CEC 2005 factories + registry dicts
    ├── utils.py         # tosz_func, tasy_func, lambda_func, rotation_matrix, penalty, etc.
    ├── tags.py          # function_characteristics metadata (BBOB)
    ├── cec2005_tags.py  # cec2005_function_characteristics metadata
    ├── bounds.py        # bbob_bounds, cec2005_bounds dicts
    └── plotting.py      # plot_2d, plot_3d (requires optional matplotlib dep)
```

### Testing

Tests live in `tests/` (`test_example.py` for BBOB, `test_cec2005.py` for CEC 2005, plus `test_bounds.py`, `test_cec2005_utils.py`). They are parametrized over all functions × multiple dimensions × both registries. Each test validates:
- Correct scalar output shape
- JIT compilation (`jax.jit`)
- Vectorization (`jax.vmap`)
- Gradient computation (`jax.grad`)

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
