from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import registry, registry_original
from bbob_jax._src.utils import _create_mesh

# Combine both registries into one parameterized source
pytest_registry = [
    pytest.param(name, fn, id=f"registry::{name}")
    for name, fn in registry.items()
]
pytest_registry_original = [
    pytest.param(name, fn, id=f"registry_original::{name}")
    for name, fn in registry_original.items()
]

all_functions = pytest_registry + pytest_registry_original


# Dimensionalities to test
dimensions = [2, 3, 5, 20, 40]


@pytest.mark.parametrize("name,fn", all_functions)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output(name, fn, dim):
    """Test that each registered function runs correctly for given dimensionalities."""
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-5.0, maxval=5.0)

    try:
        y = fn(ndim=dim, key=key)(x)
    except Exception as e:
        pytest.fail(f"Function {name} raised an exception: {e}")

    assert jnp.isfinite(y), f"Function {name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, (
        f"Function {name} did not return a scalar output: {y.shape}"
    )


@pytest.mark.parametrize("name,fn", all_functions)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output_jit(name, fn, dim):
    """Test that each registered function runs correctly for given dimensionalities."""
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-5.0, maxval=5.0)

    try:
        y = jax.jit(fn(ndim=dim, key=key))(x)
    except Exception as e:
        pytest.fail(f"Function {name} raised an exception: {e}")

    assert jnp.isfinite(y), f"Function {name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, (
        f"Function {name} did not return a scalar output: {y.shape}"
    )


@pytest.mark.parametrize("name,fn", pytest_registry)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("seed", [1, 2])
def test_function_vmap(name, fn, dim, seed):
    key = jr.key(seed)
    try:
        fn_ = fn(ndim=dim, key=key)
        _, _, Z = _create_mesh(fn_, bounds=(-5.0, 5.0), px=300)
    except Exception as e:
        pytest.fail(f"Function {name} raised an exception during vmap: {e}")

    assert jnp.all(jnp.isfinite(Z)), (
        f"Function {name} returned non-finite values in vmap output."
    )


@pytest.mark.parametrize("name,fn", pytest_registry)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("seed", [1, 2])
def test_function_grad(name, fn, dim, seed):
    key = jr.key(seed)
    key_x, key_fn = jr.split(key)
    x = jr.uniform(key_x, shape=(300, dim), minval=-5.0, maxval=5.0)
    try:
        fn_ = fn(ndim=dim, key=key_fn)
        grad_fn = jax.grad(fn_)
        grad_value = jax.vmap(grad_fn)(x)
    except Exception as e:
        pytest.fail(
            f"Function {name} raised an exception during jax.grad: {e}"
        )

    assert grad_value.shape == x.shape, (
        f"Function {name} gradient has incorrect shape: "
        f"expected {x.shape}, got {grad_value.shape}"
    )

    assert jnp.all(jnp.isfinite(grad_value)), (
        f"Function {name} returned non-finite gradient values."
    )
