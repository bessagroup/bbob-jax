"""Reference-fidelity pins for the noiseless BBOB suite (ADR 0005).

Independent inline reimplementations of the corrected pieces,
checked on deterministic instances — these regress the pre-ADR-0005
deviations (T_osz first/last-element mask and single-branch
constants, T_asy off-by-one exponent, F3/F15 multiplicative core,
F4 skew parity, F14 missing sqrt) without needing the external
reference code. The full point-for-point comparison against the
official implementation lives in
``scripts/crosscheck_bbob_noiseless.py`` (not in CI).
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from bbob_jax import problem
from bbob_jax._src.factories import _partial_keywords
from bbob_jax._src.transforms import lambda_func, tasy_func, tosz_func


def _tosz_manual(v: float) -> float:
    """Scalar reference T_osz (monotoneTFosc)."""
    if v == 0.0:
        return 0.0
    c1, c2 = (10.0, 7.9) if v > 0 else (5.5, 3.1)
    x_hat = np.log(abs(v))
    return float(
        np.sign(v)
        * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))
    )


def test_tosz_is_elementwise_with_sign_dependent_constants():
    """Every component is transformed, negatives use (5.5, 3.1)."""
    x = jnp.array([-3.7, 1.2, -0.4, 2.9, 0.8, 0.0])
    expected = jnp.array([_tosz_manual(float(v)) for v in x])
    assert jnp.allclose(tosz_func(x), expected, rtol=1e-6)
    # regression: interior elements must not pass through untouched
    assert not jnp.allclose(tosz_func(x)[1:-1], x[1:-1])


def test_tasy_exponent_is_zero_based():
    """Exponent is 1 + beta * (i/(D-1)) * sqrt(x_i), 0-based i."""
    x = jnp.array([0.5, 2.0, -1.0, 3.0])
    beta = 0.5
    idx = np.arange(4)
    expected = np.array(
        [
            v ** (1 + beta * (i / 3.0) * np.sqrt(v)) if v > 0 else v
            for i, v in zip(idx, np.asarray(x))
        ]
    )
    assert jnp.allclose(
        tasy_func(x, beta=beta), jnp.asarray(expected), rtol=1e-6
    )
    # regression: the first positive component must be unchanged
    # (exponent 1), not compressed by a negative exponent term
    assert jnp.allclose(tasy_func(x, beta=beta)[0], x[0], rtol=1e-6)


def _rastrigin_z1(x1: float) -> float:
    """First coordinate of z for the deterministic 2-D rastrigin."""
    x = jnp.array([x1, 0.0])
    lam = lambda_func(2, 10.0)
    return float((lam @ tasy_func(tosz_func(x), beta=0.2))[0])


def test_rastrigin_has_no_spurious_lattice_minimum():
    """The core is additive: at z = (1, 0) the value is 1, not f_opt.

    The pre-ADR-0005 multiplicative core made every integer lattice
    point of z a global minimum with value exactly f_opt.
    """
    p = problem("rastrigin", ndim=2, deterministic=True)
    lo, hi = 0.5, 2.0
    assert _rastrigin_z1(lo) < 1.0 < _rastrigin_z1(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _rastrigin_z1(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    x_star = jnp.array([0.5 * (lo + hi), 0.0])
    value = p.fn(x_star)
    # official value at z=(1,0): 10*(2 - cos(2pi) - cos(0)) + |z|^2 = 1
    assert jnp.isclose(value, 1.0, atol=1e-3)
    assert not jnp.isclose(value, p.f_opt, atol=1e-4)


@pytest.mark.parametrize("name", ["rastrigin", "rastrigin_seperable"])
def test_rastrigin_core_is_additive(name):
    """Away from cosine minima the value exceeds 10*(D - sum cos)."""
    p = problem(name, ndim=4, key=jr.key(0))
    x = jr.uniform(jr.key(1), (4,), minval=-4.0, maxval=4.0)
    # multiplicative core would frequently dip below the additive
    # lower bound sum(z^2) alone; pin one concrete evaluation instead
    # of a structural bound: recompute additively from the transforms.
    kw = _partial_keywords(p.fn)
    if name == "rastrigin":
        z = kw["_mat"] @ tasy_func(
            tosz_func(kw["R"] @ (x - kw["x_opt"])), beta=0.2
        )
    else:
        z = lambda_func(4, 10.0) @ tasy_func(
            tosz_func(x - kw["x_opt"]), beta=0.2
        )
    expected = (
        10.0 * (4 - jnp.sum(jnp.cos(2.0 * jnp.pi * z)))
        + jnp.sum(z**2)
        + kw["f_opt"]
    )
    assert jnp.allclose(p.fn(x), expected, rtol=1e-6)


def test_bueche_skews_even_zero_based_coordinates():
    """F4 multiplies the 0-based even coordinates by 10 when positive."""
    p = problem("skew_rastrigin_bueche", ndim=2, deterministic=True)
    x = jnp.array([0.5, 0.5])
    s = jnp.array([1.0, 10.0**0.5])
    z = s * tosz_func(x)
    z = z.at[0].set(jnp.where(z[0] > 0, 10.0 * z[0], z[0]))
    expected = 10.0 * (2 - jnp.sum(jnp.cos(2.0 * jnp.pi * z))) + jnp.sum(z**2)
    assert jnp.allclose(p.fn(x), expected, rtol=1e-5)


def test_bueche_maker_evens_optimum():
    """The reference makes even coordinates of the F4 optimum
    non-negative."""
    for seed in range(5):
        p = problem("skew_rastrigin_bueche", ndim=5, key=jr.key(seed))
        assert jnp.all(p.x_opt[::2] >= 0)


def test_sum_of_different_powers_takes_sqrt():
    p = problem("sum_of_different_powers", ndim=5, deterministic=True)
    x = jr.uniform(jr.key(3), (5,), minval=-4.0, maxval=4.0)
    idx = jnp.arange(5, dtype=x.dtype)
    expected = jnp.sqrt(jnp.sum(jnp.abs(x) ** (2 + 4 * idx / 4.0)))
    assert jnp.allclose(p.fn(x), expected, rtol=1e-6)


def test_gallagher_conditioning_exponents_are_permuted_per_peak():
    """The reference permutes each peak's conditioning diagonal."""
    p = problem("gallagher_101_peaks", ndim=6, key=jr.key(0))
    c_diags = _partial_keywords(p.fn)["_gal_c_diags"]
    unsorted_rows = jnp.sum(jnp.any(jnp.diff(c_diags, axis=1) < 0, axis=1))
    assert unsorted_rows > 50  # ascending-only rows would give 0
