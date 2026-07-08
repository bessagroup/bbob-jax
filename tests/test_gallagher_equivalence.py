"""Equivalence tests for the Gallagher rotation precompute.

The Gallagher functions rotate the peak locations once at factory time
(``y @ R.T``) and rotate only ``x`` per evaluation, instead of rotating
all ``num_peaks`` differences. This is algebraically identical
(``R @ (x - y_i) == R @ x - R @ y_i``); these tests pin values and
gradients against a reference implementation that rotates every peak
difference explicitly, as the pre-refactor code did. The reference is a
deliberately independent re-implementation and lives only in tests.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import bbob_jax
from bbob_jax._src.utils import penalty, tosz_func

GALLAGHER_PARAMS = {
    "gallagher_101_peaks": (101, 99, 1000.0, -5.0, 5.0),
    "gallagher_21_peaks": (21, 19, 1000.0**2, -4.9, 4.9),
}


def _reference_gallagher(
    x,
    x_opt,
    f_opt,
    R,
    Q,
    num_peaks,
    w_divisor,
    alpha_first,
    y_minval,
    y_maxval,
):
    """Pre-refactor Gallagher: rotate every peak difference explicitly."""
    ndim = x.shape[-1]

    key = jr.key(0)
    key = jr.fold_in(key, Q[0, 0])
    key1, key2 = jr.split(key)

    i = jnp.arange(1, num_peaks + 1, dtype=float)
    j = jnp.arange(0, num_peaks - 1, dtype=float)

    w = 1.1 + 8.0 * ((i - 2) / w_divisor)
    w = w.at[0].set(10.0)

    a = jnp.power(1000, 2.0 * (j / (num_peaks - 1)))
    alpha = jr.permutation(key1, a)
    alpha = jnp.concatenate([jnp.array([alpha_first]), alpha])

    y = jr.uniform(
        key2, shape=(num_peaks, ndim), minval=y_minval, maxval=y_maxval
    )
    y = y.at[0].set(x_opt)

    idx = jnp.arange(ndim, dtype=float)
    c_diags = jnp.power(
        alpha[:, None], idx[None, :] / (2 * (ndim - 1))
    ) / jnp.power(alpha[:, None], 0.25)

    diff = x[None, :] - y  # (num_peaks, ndim)
    rotated_diff = jnp.einsum("ij,...j->...i", R, diff)  # (num_peaks, ndim)
    exponents = -(1.0 / (2.0 * ndim)) * jnp.sum(
        c_diags * rotated_diff**2, axis=-1
    )
    inside_max = w * jnp.exp(exponents)

    f = 10.0 - jnp.max(inside_max, axis=0)
    f_tosz = tosz_func(jnp.array([f]))[0]
    return jnp.power(f_tosz, 2) + penalty(x) + f_opt


@pytest.mark.parametrize("name", list(GALLAGHER_PARAMS))
@pytest.mark.parametrize(
    "registry", [bbob_jax.registry, bbob_jax.registry_original]
)
@pytest.mark.parametrize("dim", [2, 10, 40])
def test_gallagher_matches_per_peak_rotation(name, registry, dim):
    fn, _ = registry[name](ndim=dim, key=jr.key(3))
    ref_kwargs = {k: fn.keywords[k] for k in ("x_opt", "f_opt", "R", "Q")}
    num_peaks, w_divisor, alpha_first, y_minval, y_maxval = GALLAGHER_PARAMS[
        name
    ]

    def ref(x):
        return _reference_gallagher(
            x,
            num_peaks=num_peaks,
            w_divisor=w_divisor,
            alpha_first=alpha_first,
            y_minval=y_minval,
            y_maxval=y_maxval,
            **ref_kwargs,
        )

    x = jr.uniform(jr.key(7), (32, dim), minval=-5.0, maxval=5.0)

    value_new = jax.vmap(fn)(x)
    value_ref = jax.vmap(ref)(x)
    assert jnp.allclose(value_new, value_ref, rtol=1e-4, atol=1e-5), (
        f"{name} (dim={dim}): max value deviation "
        f"{jnp.max(jnp.abs(value_new - value_ref))} from the per-peak "
        f"rotation reference"
    )

    grad_new = jax.vmap(jax.grad(fn))(x)
    grad_ref = jax.vmap(jax.grad(ref))(x)
    assert jnp.allclose(grad_new, grad_ref, rtol=1e-3, atol=1e-4), (
        f"{name} (dim={dim}): max gradient deviation "
        f"{jnp.max(jnp.abs(grad_new - grad_ref))} from the per-peak "
        f"rotation reference"
    )
