"""Tests for the Problem accessor.

``problem(name, ndim, key)`` bundles fn, x_opt, f_opt, bounds,
tags and noise arity in one lookup. The optimum checks here
are the package's invariant that ``x_opt`` really is the
argmin: fn(x_opt) == f_opt for every function in both suites
(with documented exceptions for the deterministic CEC
compositions, whose weighting is degenerate).
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import (
    Problem,
    bbob_bounds,
    cec2005_bounds,
    cec2005_function_characteristics,
    cec2005_registry,
    cec2017_bounds,
    cec2017_function_characteristics,
    cec2017_registry,
    function_characteristics,
    problem,
    registry,
)

BBOB_NAMES = list(registry.keys())
CEC_NAMES = list(cec2005_registry.keys())
CEC2017_NAMES = list(cec2017_registry.keys())
COMPOSITION_NAMES = [f"f{i}" for i in range(15, 26)]

_NOISY = {
    name
    for name, chars in cec2005_function_characteristics.items()
    if chars["noise"]
}
_STRUCTURE_MODIFIED = {
    name
    for name, chars in cec2005_function_characteristics.items()
    if chars["structure_modified"]
}


def _evaluate_at_optimum(p: Problem) -> jnp.ndarray:
    if p.noisy:
        return p.fn(p.x_opt, jr.key(42))
    return p.fn(p.x_opt)


def test_problem_is_named_tuple():
    p = problem("sphere", ndim=2, key=jr.key(0))
    assert isinstance(p, Problem)
    assert p.name == "sphere"


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        problem("not_a_function", ndim=2, key=jr.key(0))


def test_missing_key_raises():
    with pytest.raises(ValueError):
        problem("sphere", ndim=2)


@pytest.mark.parametrize("name", BBOB_NAMES + CEC_NAMES + CEC2017_NAMES)
def test_problem_fields_match_metadata_dicts(name):
    """Problem bundles the same facts the separate dicts expose."""
    p = problem(name, ndim=3, key=jr.key(0))
    if name in BBOB_NAMES:
        assert p.bounds == bbob_bounds[name]
        assert p.tags == function_characteristics[name]
        assert p.noisy is False
    elif name in CEC_NAMES:
        assert p.bounds == cec2005_bounds[name]
        assert p.tags == cec2005_function_characteristics[name]
        assert p.noisy == cec2005_function_characteristics[name]["noise"]
    else:
        assert p.bounds == cec2017_bounds[name]
        assert p.tags == cec2017_function_characteristics[name]
        assert p.noisy is False
        assert p.min_ndim >= 1


@pytest.mark.parametrize("name", BBOB_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 20])
@pytest.mark.parametrize("seed", [0, 1])
def test_problem_fn_matches_registry(name, dim, seed):
    """The registry tuple is an adapter of the same instance."""
    key = jr.key(seed)
    p = problem(name, ndim=dim, key=key)
    fn, f_opt = registry[name](ndim=dim, key=key)
    x = jr.uniform(jr.key(99), shape=(dim,), minval=-5.0, maxval=5.0)
    assert jnp.array_equal(p.f_opt, f_opt)
    assert jnp.array_equal(p.fn(x), fn(x))


@pytest.mark.parametrize("name", BBOB_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 20, 40])
@pytest.mark.parametrize("deterministic", [False, True])
def test_bbob_global_minimum_at_x_opt(name, dim, deterministic):
    """fn(x_opt) == f_opt for every BBOB function, both modes.

    The 1e-5 tolerance covers float32 rounding (the worst
    observed residual is ~1e-6 for griewank_rosenbrock_f8f2 in
    deterministic mode); everything else lands within 1e-8.
    """
    p = problem(name, ndim=dim, key=jr.key(0), deterministic=deterministic)
    result = p.fn(p.x_opt)
    assert jnp.isclose(result, p.f_opt, atol=1e-5), (
        f"{name} dim={dim} deterministic={deterministic}: "
        f"fn(x_opt)={result}, f_opt={p.f_opt}, diff={result - p.f_opt}"
    )


@pytest.mark.parametrize("name", CEC_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 10])
def test_cec2005_global_minimum_at_x_opt(name, dim):
    """fn(x_opt) == f_opt for every CEC 2005 function (randomized).

    Tolerances mirror test_cec2005.test_global_minimum_at_xopt:
    soft rounding shifts the structure-modified functions'
    effective minimum slightly, and the composition weighting
    is a smooth approximation.
    """
    p = problem(name, ndim=dim, key=jr.key(0))
    result = _evaluate_at_optimum(p)

    if name in _STRUCTURE_MODIFIED:
        atol = 5e-2
    elif name in COMPOSITION_NAMES:
        atol = 1e-2
    else:
        atol = 1e-8

    assert jnp.isclose(result, p.f_opt, atol=atol), (
        f"{name} dim={dim}: fn(x_opt)={result}, f_opt={p.f_opt}, "
        f"diff={result - p.f_opt}"
    )


@pytest.mark.parametrize("name", CEC_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 10])
def test_cec2005_global_minimum_at_x_opt_deterministic(name, dim):
    """Deterministic CEC instances reach f_opt at x_opt.

    Compositions (F15-F25) are skipped: with all component
    optima at zero the weighting is degenerate, as documented
    on ``Problem.x_opt``.
    """
    if name in COMPOSITION_NAMES:
        pytest.skip("Deterministic compositions are degenerate")

    p = problem(name, ndim=dim, deterministic=True)
    result = _evaluate_at_optimum(p)
    assert jnp.isclose(result, p.f_opt, atol=1e-8), (
        f"{name} dim={dim}: fn(x_opt)={result}, f_opt={p.f_opt}, "
        f"diff={result - p.f_opt}"
    )


@pytest.mark.parametrize("name", ["f4", "f17", "f24", "f25"])
def test_noisy_problems_take_key(name):
    """Noisy problems advertise their fn(x, key) arity."""
    p = problem(name, ndim=3, key=jr.key(0))
    assert p.noisy
    value = p.fn(jnp.zeros(3), jr.key(1))
    assert jnp.isfinite(value)


@pytest.mark.parametrize("name", CEC2017_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 10])
@pytest.mark.parametrize("deterministic", [False, True])
def test_cec2017_global_minimum_at_x_opt(name, dim, deterministic):
    """fn(x_opt) == f_opt for every CEC 2017 function, both modes.

    The 1e-5 tolerance covers the epsilon guards inside the kernels'
    square roots (F6's Schaffer F7 floor is ~1e-6); for ``cec2017_f9``
    the resolver places x_opt at the rotated all-ones point, so this
    also pins the Levy optimum-displacement handling.
    """
    p = problem(name, ndim=dim, key=jr.key(0), deterministic=deterministic)
    result = p.fn(p.x_opt)
    assert jnp.isclose(result, p.f_opt, atol=1e-5), (
        f"{name} dim={dim} deterministic={deterministic}: "
        f"fn(x_opt)={result}, f_opt={p.f_opt}, diff={result - p.f_opt}"
    )


def test_cec2017_min_ndim_surfaces_in_problem():
    p = problem("cec2017_f6", ndim=2, key=jr.key(0))
    assert p.min_ndim == 2
    with pytest.raises(ValueError, match="ndim >= 2"):
        problem("cec2017_f6", ndim=1, key=jr.key(0))
