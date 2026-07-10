"""Tests for CEC 2017 benchmark functions (F1, F3-F10).

These tests validate JAX compatibility (jit, vmap, grad), NaN propagation
and basic sanity checks. They do NOT validate that results match the
official CEC 2017 data files — instances are seed-generated (see the
``cec2017`` module docstring); a cross-validation script against the
official reference data accompanies the composition wave.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import cec2017_registry, cec2017_registry_original
from bbob_jax._src.mesh import _create_mesh

pytest_cec2017 = [
    pytest.param(name, fn, id=f"cec2017_registry::{name}")
    for name, fn in cec2017_registry.items()
]
pytest_cec2017_original = [
    pytest.param(name, fn, id=f"cec2017_registry_original::{name}")
    for name, fn in cec2017_registry_original.items()
]
all_cec2017 = pytest_cec2017 + pytest_cec2017_original

dimensions = [2, 3, 5, 10, 20]


@pytest.mark.parametrize("name,fn", all_cec2017)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-100.0, maxval=100.0)
    fn_func, _ = fn(ndim=dim, key=key)
    y = fn_func(x)
    assert jnp.isfinite(y), f"{name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, f"{name} did not return scalar: {y.shape}"


@pytest.mark.parametrize("name,fn", all_cec2017)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output_jit(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-100.0, maxval=100.0)
    fn_func, _ = fn(ndim=dim, key=key)
    y = jax.jit(fn_func)(x)
    assert jnp.isfinite(y), f"{name} JIT returned non-finite: {y}"
    assert jnp.ndim(y) == 0


@pytest.mark.parametrize("name,fn", all_cec2017)
@pytest.mark.parametrize("dim", dimensions)
def test_function_nan_propagates(name, fn, dim):
    """NaN inputs must propagate to the output, not be silently masked.

    Same guarantee as the BBOB suite (see test_example.py): a function
    that returns a finite value for a NaN input hides invalid inputs
    from the caller. This is why the CEC 2017 kernels use
    epsilon-guarded ``jnp.sqrt`` instead of ``sj.sqrt`` (which maps
    NaN to 0).
    """
    key = jr.key(0)
    fn_func, _ = fn(ndim=dim, key=key)

    x_all_nan = jnp.full((dim,), jnp.nan)
    assert jnp.isnan(fn_func(x_all_nan)), (
        f"Function {name} did not propagate an all-NaN input."
    )

    x_partial_nan = jnp.zeros((dim,)).at[0].set(jnp.nan)
    assert jnp.isnan(fn_func(x_partial_nan)), (
        f"Function {name} did not propagate a single-coordinate NaN input."
    )

    x_last_nan = jnp.zeros((dim,)).at[-1].set(jnp.nan)
    assert jnp.isnan(fn_func(x_last_nan)), (
        f"Function {name} did not propagate a last-coordinate NaN input."
    )


@pytest.mark.parametrize("name,fn", pytest_cec2017)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("seed", [1, 2])
def test_function_vmap(name, fn, dim, seed):
    key = jr.key(seed)
    fn_func, _ = fn(ndim=dim, key=key)
    _, _, Z = _create_mesh(fn_func, bounds=(-100.0, 100.0), px=50)
    assert jnp.all(jnp.isfinite(Z)), f"{name} vmap returned non-finite values"


@pytest.mark.parametrize("name,fn", pytest_cec2017)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("seed", [1, 2])
def test_function_grad(name, fn, dim, seed):
    key = jr.key(seed)
    key_x, key_fn = jr.split(key)
    x = jr.uniform(key_x, shape=(10, dim), minval=-100.0, maxval=100.0)
    fn_func, _ = fn(ndim=dim, key=key_fn)
    grad_value = jax.vmap(jax.grad(fn_func))(x)
    assert grad_value.shape == x.shape
    assert jnp.all(jnp.isfinite(grad_value)), (
        f"{name} grad returned non-finite"
    )


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(cec2017_registry["cec2017_f6"], id="randomized"),
        pytest.param(cec2017_registry_original["cec2017_f6"], id="original"),
    ],
)
def test_f6_min_ndim_raises(fn):
    """Schaffer F7's ``1/(D-1)^2`` normalization is undefined at D=1."""
    with pytest.raises(ValueError, match="ndim >= 2"):
        fn(ndim=1, key=jr.key(0))


@pytest.mark.parametrize("dim", [3, 10])
def test_f5_f8_same_structure_different_instance(dim):
    """F8's non-continuity transform is dead code in the reference,
    so F5 and F8 share their math but not their sampled instance."""
    key = jr.key(7)
    f5_fn, _ = cec2017_registry["cec2017_f5"](ndim=dim, key=key)
    f8_fn, _ = cec2017_registry["cec2017_f8"](ndim=dim, key=key)
    x = jr.uniform(jr.key(8), shape=(dim,), minval=-100.0, maxval=100.0)
    # Identical math with identical parameters (same key, same maker) ...
    assert jnp.array_equal(f5_fn(x), f8_fn(x))
    # ... but downstream instances use per-function keys, and the
    # deterministic instances coincide by construction.
    f5_det, _ = cec2017_registry_original["cec2017_f5"](ndim=dim)
    f8_det, _ = cec2017_registry_original["cec2017_f8"](ndim=dim)
    assert jnp.array_equal(f5_det(x), f8_det(x))


@pytest.mark.parametrize("name,fn", pytest_cec2017_original)
def test_deterministic_needs_no_key(name, fn):
    fn_func, f_opt = fn(ndim=3)
    assert jnp.isfinite(fn_func(jnp.ones(3)))
    assert f_opt == 0.0
