import jax
import jax.numpy as jnp
import pytest

from bbob_jax._src.utils import (
    ackley,
    cec2005_weierstrass,
    griewank,
    scaffer_f6,
)


def test_ackley_minimum():
    """Ackley minimum is approximately 0 at origin.

    A small epsilon (1e-10) is added under the sqrt to avoid NaN gradients at
    x=0 (sqrt'(0) = inf). This shifts the function value by ~4e-5 at x=0.
    """
    x = jnp.zeros(5)
    assert jnp.isclose(ackley(x), 0.0, atol=1e-4)


def test_ackley_shape():
    x = jnp.ones(10)
    assert ackley(x).shape == ()


def test_griewank_minimum():
    """Griewank minimum is 0 at origin."""
    x = jnp.zeros(5)
    assert jnp.isclose(griewank(x), 0.0, atol=1e-10)


def test_griewank_shape():
    x = jnp.ones(10)
    assert griewank(x).shape == ()


def test_scaffer_f6_minimum():
    """Scaffer F6 minimum is 0 at origin."""
    assert jnp.isclose(
        scaffer_f6(jnp.array(0.0), jnp.array(0.0)), 0.0, atol=1e-10
    )


def test_scaffer_f6_shape():
    assert scaffer_f6(jnp.array(1.0), jnp.array(2.0)).shape == ()


def test_cec2005_weierstrass_minimum():
    """Weierstrass minimum is 0 at origin (by definition of the constant subtraction)."""
    x = jnp.zeros(5)
    assert jnp.isclose(cec2005_weierstrass(x), 0.0, atol=1e-5)


def test_cec2005_weierstrass_shape():
    x = jnp.ones(10)
    assert cec2005_weierstrass(x).shape == ()


@pytest.mark.parametrize("fn", [ackley, griewank, cec2005_weierstrass])
def test_utils_grad_compatible(fn):
    x = jnp.ones(5)
    grad = jax.grad(fn)(x)
    assert grad.shape == (5,)
    assert jnp.all(jnp.isfinite(grad))
