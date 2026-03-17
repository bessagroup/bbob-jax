"""Tests for CEC 2005 benchmark functions.

These tests validate JAX compatibility (jit, vmap, grad) and basic sanity
checks. They do NOT validate that results match official CEC 2005 data files —
see docs/superpowers/specs/2026-03-17-cec2005-jax-design.md for the rationale.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import cec2005_registry, cec2005_registry_original
from bbob_jax._src.utils import _create_mesh

pytest_cec2005 = [
    pytest.param(name, fn, id=f"cec2005_registry::{name}")
    for name, fn in cec2005_registry.items()
]
pytest_cec2005_original = [
    pytest.param(name, fn, id=f"cec2005_registry_original::{name}")
    for name, fn in cec2005_registry_original.items()
]
all_cec2005 = pytest_cec2005 + pytest_cec2005_original

dimensions = [2, 3, 5, 10, 20]

COMPOSITION_NAMES = [f"f{i}" for i in range(15, 26)]


@pytest.mark.parametrize("name,fn", all_cec2005)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-100.0, maxval=100.0)
    try:
        fn_func, _ = fn(ndim=dim, key=key)
        y = fn_func(x)
    except NotImplementedError:
        pytest.skip(f"Function {name} not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} raised an exception: {e}")
    assert jnp.isfinite(y), f"{name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, f"{name} did not return scalar: {y.shape}"


@pytest.mark.parametrize("name,fn", all_cec2005)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output_jit(name, fn, dim):
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-100.0, maxval=100.0)
    try:
        fn_func, _ = fn(ndim=dim, key=key)
        y = jax.jit(fn_func)(x)
    except NotImplementedError:
        pytest.skip(f"Function {name} JIT not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} JIT raised an exception: {e}")
    assert jnp.isfinite(y), f"{name} JIT returned non-finite: {y}"
    assert jnp.ndim(y) == 0


@pytest.mark.parametrize("name,fn", pytest_cec2005)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("seed", [1, 2])
def test_function_vmap(name, fn, dim, seed):
    key = jr.key(seed)
    try:
        fn_func, _ = fn(ndim=dim, key=key)
        _, _, Z = _create_mesh(fn_func, bounds=(-100.0, 100.0), px=50)
    except NotImplementedError:
        pytest.skip(f"Function {name} vmap not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} vmap raised an exception: {e}")
    assert jnp.all(jnp.isfinite(Z)), f"{name} vmap returned non-finite values"


@pytest.mark.parametrize("name,fn", pytest_cec2005)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("seed", [1, 2])
def test_function_grad(name, fn, dim, seed):
    key = jr.key(seed)
    key_x, key_fn = jr.split(key)
    x = jr.uniform(key_x, shape=(10, dim), minval=-100.0, maxval=100.0)
    try:
        fn_func, _ = fn(ndim=dim, key=key_fn)
        grad_fn = jax.grad(fn_func)
        grad_value = jax.vmap(grad_fn)(x)
    except NotImplementedError:
        pytest.skip(f"Function {name} grad not yet implemented")
    except Exception as e:
        pytest.fail(f"Function {name} grad raised an exception: {e}")
    assert grad_value.shape == x.shape
    assert jnp.all(jnp.isfinite(grad_value)), (
        f"{name} grad returned non-finite"
    )


@pytest.mark.parametrize("name", COMPOSITION_NAMES)
@pytest.mark.parametrize("dim", [2, 5, 10])
def test_composition_sanity(name, dim):
    """With deterministic registry, f(zeros) is finite and gradient is small."""
    fn_factory = cec2005_registry_original[name]
    fn_func, _ = fn_factory(ndim=dim, key=jr.key(0))
    x_test = jnp.zeros(dim)
    try:
        result = fn_func(x_test)
        assert jnp.isfinite(result), f"{name} sanity: non-finite at zeros"
        grad = jax.grad(fn_func)(x_test)
        grad_norm = jnp.linalg.norm(grad)
        assert grad_norm < dim * 10.0, (
            f"{name} sanity: grad norm {grad_norm:.2f} >= {dim * 10.0}"
        )
    except NotImplementedError:
        pytest.skip(f"{name} not yet implemented")
    except Exception as e:
        pytest.fail(f"{name} sanity raised an exception: {e}")
