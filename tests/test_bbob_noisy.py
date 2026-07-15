"""Tests for the BBOB-noisy suite (f101-f130).

JAX-compatibility battery (jit, vmap, grad, NaN propagation)
mirroring the other suites, plus noise-specific checks: the
pinned-draw formula tests re-derive every function's disturbed
value from its ``fn_true`` residual and the noise-model formulas
(same key-split protocol), and the statistical tests pin the
three models' distributions with a fixed key set (fully
deterministic, no flakiness). The deterministic undisturbed path
is additionally cross-checked against the compiled legacy C
reference by ``scripts/crosscheck_bbob_noisy.py`` (not in CI).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import (
    bbob_noisy_function_characteristics,
    bbob_noisy_registry,
    bbob_noisy_registry_original,
    problem,
)
from bbob_jax._src.mesh import _create_mesh
from bbob_jax._src.noise import TOL

pytest_noisy = [
    pytest.param(name, fn, id=f"bbob_noisy_registry::{name}")
    for name, fn in bbob_noisy_registry.items()
]
pytest_noisy_original = [
    pytest.param(name, fn, id=f"bbob_noisy_registry_original::{name}")
    for name, fn in bbob_noisy_registry_original.items()
]
all_noisy = pytest_noisy + pytest_noisy_original

NAMES = list(bbob_noisy_registry.keys())

dimensions = [2, 3, 5, 10, 20]


def _noise_params(name: str, ndim: int) -> tuple[str, dict]:
    """Model name and parameters implied by a function's tags."""
    tags = bbob_noisy_function_characteristics[name]
    severe = tags["severe"]
    if tags["gaussian_noise"]:
        return "gauss", {"beta": 1.0 if severe else 0.01}
    if tags["uniform_noise"]:
        alpha = 0.49 + 1.0 / ndim
        if not severe:
            alpha *= 0.01
        return "uniform", {"alpha": alpha, "beta": 1.0 if severe else 0.01}
    return "cauchy", (
        {"alpha": 1.0, "p": 0.2} if severe else {"alpha": 0.01, "p": 0.05}
    )


def _expected_disturbed(
    model: str, params: dict, residual: jax.Array, key: jax.Array
) -> jax.Array:
    """Reference reimplementation of the noise models (gate included)."""
    if model == "gauss":
        f_val = residual * jnp.exp(params["beta"] * jr.normal(key))
    elif model == "uniform":
        key1, key2 = jr.split(key)
        u1, u2 = jr.uniform(key1), jr.uniform(key2)
        f_val = (
            u1 ** params["beta"]
            * residual
            * jnp.maximum(
                1.0, (1e9 / (residual + 1e-99)) ** (params["alpha"] * u2)
            )
        )
    else:
        key1, key2, key3 = jr.split(key, 3)
        n1, n2 = jr.normal(key1), jr.normal(key2)
        u = jr.uniform(key3)
        f_val = residual + params["alpha"] * jnp.maximum(
            0.0, 1e3 + (u < params["p"]) * n1 / jnp.abs(n2 + 1e-199)
        )
    return jnp.where(residual < TOL, residual, f_val + 1.01 * TOL)


#                                                     JAX-compatibility battery
# =============================================================================


@pytest.mark.parametrize("name,fn", all_noisy)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-5.0, maxval=5.0)
    fn_func, _ = fn(ndim=dim, key=key)
    y = fn_func(x, jr.key(42))
    assert jnp.isfinite(y), f"{name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, f"{name} did not return scalar: {y.shape}"


@pytest.mark.parametrize("name,fn", all_noisy)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output_jit(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-5.0, maxval=5.0)
    fn_func, _ = fn(ndim=dim, key=key)
    y = jax.jit(fn_func)(x, jr.key(42))
    assert jnp.isfinite(y), f"{name} JIT returned non-finite: {y}"
    assert jnp.ndim(y) == 0


@pytest.mark.parametrize("name,fn", pytest_noisy)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("seed", [1, 2])
def test_function_vmap(name, fn, dim, seed):
    key = jr.key(seed)
    fn_func, _ = fn(ndim=dim, key=key)
    _, _, Z = _create_mesh(
        lambda x: fn_func(x, jr.key(42)), bounds=(-5.0, 5.0), px=50
    )
    assert jnp.all(jnp.isfinite(Z)), f"{name} vmap returned non-finite"


@pytest.mark.parametrize("name,fn", pytest_noisy)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("seed", [1, 2])
def test_function_grad(name, fn, dim, seed):
    key = jr.key(seed)
    key_x, key_fn = jr.split(key)
    x = jr.uniform(key_x, shape=(10, dim), minval=-5.0, maxval=5.0)
    fn_func, _ = fn(ndim=dim, key=key_fn)
    grad_fn = jax.grad(lambda x: fn_func(x, jr.key(42)))
    grad_value = jax.vmap(grad_fn)(x)
    assert grad_value.shape == x.shape
    assert jnp.all(jnp.isfinite(grad_value)), (
        f"{name} grad returned non-finite"
    )


@pytest.mark.parametrize("name,fn", all_noisy)
def test_nan_propagation(name, fn):
    key = jr.key(0)
    fn_func, _ = fn(ndim=3, key=key)
    y = fn_func(jnp.full(3, jnp.nan), jr.key(42))
    assert jnp.isnan(y), f"{name} did not propagate NaN: {y}"


#                                                          Noise-specific tests
# =============================================================================


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 1])
def test_key_determinism(name, seed):
    """Same key -> same value; a key batch is not constant."""
    p = problem(name, ndim=3, key=jr.key(seed))
    x = jr.uniform(jr.key(9), shape=(3,), minval=-4.0, maxval=4.0)
    v1 = p.fn(x, jr.key(5))
    v2 = p.fn(x, jr.key(5))
    assert jnp.array_equal(v1, v2)
    # seldom Cauchy noise adds a constant offset for most keys, so
    # uniqueness needs a batch, not a pair (256 keys -> the outlier
    # branch triggers with overwhelming probability).
    keys = jr.split(jr.key(6), 256)
    values = jax.vmap(lambda k: p.fn(x, k))(keys)
    assert jnp.unique(values).size > 1, f"{name} looks noise-free"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dim", [2, 5])
def test_disturbed_value_matches_noise_formula(name, dim):
    """fn(x, key) == noise_model(fn_true residual) + penalty + f_opt.

    Re-derives the disturbed value from the undisturbed residual
    with the documented formulas and key-split protocol — pins
    the noise wiring (model choice, severity parameters, gate,
    what the noise does and does not disturb) for all 30
    functions.
    """
    p = problem(name, ndim=dim, key=jr.key(3))
    # in-bounds x: no boundary penalty term
    x = jr.uniform(jr.key(8), shape=(dim,), minval=-4.0, maxval=4.0)
    residual = p.fn_true(x) - p.f_opt
    model, params = _noise_params(name, dim)
    key = jr.key(21)
    expected = _expected_disturbed(model, params, residual, key) + p.f_opt
    actual = p.fn(x, key)
    # rtol 1e-5: the uniform model's (1e9/residual)**(alpha*u2) amplifies
    # float32 last-ulp differences in the residual to ~1e-6 relative.
    assert jnp.allclose(actual, expected, rtol=1e-5), (
        f"{name}: expected {expected}, got {actual}"
    )


def test_penalty_outside_noise():
    """The x100 boundary penalty is added outside the noise.

    On the sphere the residual is computable by hand, so both
    the undisturbed and the disturbed value at an out-of-bounds
    point can be checked exactly: the penalty term appears
    undamped in both."""
    p = problem("bbob_noisy_f107", ndim=3, key=jr.key(2))
    x_out = jnp.array([6.0, -7.0, 0.5])
    excess = jnp.sum(jnp.maximum(jnp.abs(x_out) - 5.0, 0.0) ** 2)
    residual = jnp.sum((x_out - p.x_opt) ** 2)
    assert jnp.allclose(
        p.fn_true(x_out), residual + 100.0 * excess + p.f_opt, rtol=1e-6
    )
    key = jr.key(4)
    disturbed = _expected_disturbed("gauss", {"beta": 1.0}, residual, key)
    assert jnp.allclose(
        p.fn(x_out, key),
        disturbed + 100.0 * excess + p.f_opt,
        rtol=1e-6,
    )


def test_gate_returns_undisturbed_below_tol():
    """Residuals below TOL bypass the noise entirely."""
    p = problem("bbob_noisy_f107", ndim=4, key=jr.key(0))
    keys = jr.split(jr.key(1), 32)
    values = jax.vmap(lambda k: p.fn(p.x_opt, k))(keys)
    assert jnp.all(values == p.f_opt)


def test_gaussian_noise_statistics():
    """f107: log((fval - f_opt - 1.01e-8) / residual) ~ N(0, 1)."""
    p = problem("bbob_noisy_f107", ndim=5, key=jr.key(0))
    x = jr.uniform(jr.key(2), shape=(5,), minval=-4.0, maxval=4.0)
    residual = p.fn_true(x) - p.f_opt
    keys = jr.split(jr.key(3), 4096)
    values = jax.vmap(lambda k: p.fn(x, k))(keys)
    z = jnp.log((values - p.f_opt - 1.01 * TOL) / residual)
    assert jnp.abs(jnp.mean(z)) < 0.1
    assert jnp.abs(jnp.std(z) - 1.0) < 0.1


def test_cauchy_noise_statistics():
    """f109: the Cauchy outlier triggers with probability p = 0.2."""
    p = problem("bbob_noisy_f109", ndim=5, key=jr.key(0))
    x = jr.uniform(jr.key(2), shape=(5,), minval=-4.0, maxval=4.0)
    residual = p.fn_true(x) - p.f_opt
    keys = jr.split(jr.key(3), 4096)
    values = jax.vmap(lambda k: p.fn(x, k))(keys)
    extra = values - p.f_opt - 1.01 * TOL - residual
    # non-outlier draws add exactly alpha * 1e3
    outlier_frac = jnp.mean(jnp.abs(extra - 1e3) > 1e-2)
    assert 0.15 < outlier_frac < 0.25


def test_uniform_noise_statistics():
    """f108: support and amplification of the uniform model."""
    p = problem("bbob_noisy_f108", ndim=5, key=jr.key(0))
    x = jr.uniform(jr.key(2), shape=(5,), minval=-4.0, maxval=4.0)
    residual = p.fn_true(x) - p.f_opt
    keys = jr.split(jr.key(3), 4096)
    values = jax.vmap(lambda k: p.fn(x, k))(keys)
    disturbed = values - p.f_opt - 1.01 * TOL
    alpha = 0.49 + 1.0 / 5
    upper = residual * (1e9 / residual) ** alpha
    assert jnp.all(disturbed > 0.0)
    assert jnp.all(disturbed <= upper * 1.001)
    # amplification dominates attenuation away from the optimum
    assert jnp.mean(disturbed > residual) > 0.5


def test_deterministic_instances_are_zero_at_origin_optimum():
    for name in ("bbob_noisy_f101", "bbob_noisy_f113", "bbob_noisy_f128"):
        p = problem(name, ndim=3, deterministic=True)
        assert p.fn(p.x_opt, jr.key(0)) == 0.0
        assert p.fn_true(p.x_opt) == 0.0
