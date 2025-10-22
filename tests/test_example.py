import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bbob_jax import registry, registry_original

# Combine both registries into one parameterized source
all_functions = [
    pytest.param(name, fn, id=f"registry::{name}")
    for name, fn in registry.items()
] + [
    pytest.param(name, fn, id=f"registry_original::{name}")
    for name, fn in registry_original.items()
]

# Dimensionalities to test
dimensions = [2, 3, 5, 20, 40]


@pytest.mark.parametrize("name,fn", all_functions)
@pytest.mark.parametrize("dim", dimensions)
def test_function_output(name, fn, dim):
    """Test that each registered function runs correctly for given dimensionalities."""
    key = jr.key(0)
    x = jr.uniform(key, shape=(dim,), minval=-5.0, maxval=5.0)

    try:
        y = fn(x, key=key)
    except Exception as e:
        pytest.fail(f"Function {name} raised an exception: {e}")

    assert jnp.isfinite(y), f"Function {name} returned non-finite value: {y}"
    assert jnp.ndim(y) == 0, (
        f"Function {name} did not return a scalar output: {y.shape}"
    )
