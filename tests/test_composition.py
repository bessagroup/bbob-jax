import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbob_jax._src.composition import (
    ackley,
    cec2005_weierstrass,
    griewank,
    hybrid_composition,
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


# --- hybrid_composition: the CEC 2005 F15-F25 weights -----------------------

_N_COMP = 10
_NDIM = 4


def _def2c_weights(x, x_opt, sigma):
    """Port of ``calc_weight`` in the official CEC 2005 ``src/def2.c``.

    Every weight except the largest is damped by ``1 - w_max**10``, then
    all are divided by their sum (the sum taken after damping). Uniform
    weights when that sum is 0.
    """
    ndim = x.shape[-1]
    w = np.exp(-np.sum((x - x_opt) ** 2, axis=1) / (2.0 * ndim * sigma**2))
    w_max = w.max()
    w = np.where(w != w_max, w * (1.0 - w_max**10), w)
    total = w.sum()
    if total == 0.0:
        return np.full_like(w, 1.0 / w.size)
    return w / total


def _instance(seed):
    rng = np.random.default_rng(seed)
    x_opt = rng.uniform(-2.0, 2.0, size=(_N_COMP, _NDIM))
    sigma = rng.uniform(0.5, 2.0, size=_N_COMP)
    lambda_ = rng.uniform(0.5, 2.0, size=_N_COMP)
    M = np.stack(
        [
            np.linalg.qr(rng.normal(size=(_NDIM, _NDIM)))[0]
            for _ in range(_N_COMP)
        ]
    )
    f_max = rng.uniform(10.0, 100.0, size=_N_COMP)
    bias = 100.0 * np.arange(_N_COMP)
    return x_opt, sigma, lambda_, M, f_max, bias


def _sphere(z):
    return jnp.sum(z**2)


def _expected(x, x_opt, sigma, lambda_, M, f_max, bias, C=2000.0):
    fits = np.array(
        [
            np.sum((M[i] @ ((x - x_opt[i]) / lambda_[i])) ** 2)
            for i in range(_N_COMP)
        ]
    )
    w = _def2c_weights(x, x_opt, sigma)
    return float(np.sum(w * (C * fits / f_max + bias)))


def _compose(x, x_opt, sigma, lambda_, M, f_max, bias):
    return hybrid_composition(
        jnp.asarray(x),
        [_sphere] * _N_COMP,
        jnp.asarray(sigma),
        jnp.asarray(lambda_),
        jnp.asarray(bias),
        jnp.asarray(x_opt),
        jnp.asarray(M),
        _f_max=jnp.asarray(f_max),
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("offset", [0.05, 0.3, 1.0, 3.0])
def test_hybrid_composition_weights_match_def2c(seed, offset):
    """Weights as the official ``calc_weight``: damp, then normalize.

    Points at several distances from one component's optimum, so the
    damping factor ``1 - w_max**10`` runs from near 0 to near 1.
    """
    inst = _instance(seed)
    rng = np.random.default_rng(100 + seed)
    x = inst[0][seed % _N_COMP] + offset * rng.normal(size=_NDIM)
    with jax.enable_x64(True):
        got = float(_compose(x, *inst))
    assert got == pytest.approx(_expected(x, *inst), rel=1e-12, abs=1e-9)


def test_hybrid_composition_at_a_component_optimum():
    """At ``o_k`` the damping is total: only component k counts."""
    inst = _instance(0)
    x_opt, bias = inst[0], inst[5]
    with jax.enable_x64(True):
        for k in (0, 3, 9):
            assert float(_compose(x_opt[k], *inst)) == pytest.approx(
                bias[k], abs=1e-9
            )


def test_hybrid_composition_weights_do_not_underflow():
    """Far from every optimum each ``exp`` underflows to 0 in float64.

    The official code then falls back to uniform weights. Computed in
    log space, the weights stay a proper softmax of ``-d^2 / (2 n
    sigma^2)``, dominated by the nearest component (relative to sigma).
    """
    x_opt, sigma, lambda_, M, f_max, bias = _instance(1)
    x = np.full(_NDIM, 400.0)
    log_w = -np.sum((x - x_opt) ** 2, axis=1) / (2.0 * _NDIM * sigma**2)
    assert np.all(np.exp(log_w) == 0.0)  # the plain weights underflow
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    fits = np.array(
        [
            np.sum((M[i] @ ((x - x_opt[i]) / lambda_[i])) ** 2)
            for i in range(_N_COMP)
        ]
    )
    expected = float(np.sum(w * (2000.0 * fits / f_max + bias)))
    with jax.enable_x64(True):
        got = float(_compose(x, x_opt, sigma, lambda_, M, f_max, bias))
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("where", ["optimum", "near", "far"])
def test_hybrid_composition_grad_is_finite(where):
    inst = _instance(2)
    x = {
        "optimum": inst[0][4],
        "near": inst[0][4] + 0.1,
        "far": np.full(_NDIM, 400.0),
    }[where]
    with jax.enable_x64(True):
        grad = jax.grad(lambda v: _compose(v, *inst))(jnp.asarray(x))
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("offset", [0.05, 0.3, 1.0, 3.0])
def test_hybrid_composition_grad_matches_finite_differences(seed, offset):
    """``jax.grad`` is the true derivative, damping included.

    Smooth (sphere) components, at distances from one optimum where the
    damping ``1 - w_max**10`` lies strictly between 0 and 1, so a
    gradient that ignored its dependence on ``x`` would show here.
    """
    inst = _instance(seed)
    rng = np.random.default_rng(200 + seed)
    x0 = inst[0][seed % _N_COMP] + offset * rng.normal(size=_NDIM)
    h = 1e-6
    with jax.enable_x64(True):
        f = jax.jit(lambda v: _compose(v, *inst))
        grad = np.asarray(jax.grad(lambda v: _compose(v, *inst))(x0))
        fd = np.array(
            [
                (float(f(x0 + h * e)) - float(f(x0 - h * e))) / (2 * h)
                for e in np.eye(_NDIM)
            ]
        )
    np.testing.assert_allclose(
        grad, fd, rtol=1e-5, atol=1e-5 * max(1.0, np.max(np.abs(fd)))
    )


@pytest.mark.parametrize("gap", [0.0, 1e-4, 1e-2])
def test_hybrid_composition_near_ties_pick_the_exact_largest(gap):
    """Two nearly equidistant components, as ``calc_weight`` decides.

    Only the exact largest weight escapes the damping (an exact tie
    keeps both), however close the runner-up is.
    """
    x_opt, sigma, lambda_, M, f_max, bias = _instance(3)
    sigma = np.ones(_N_COMP)
    x = np.zeros(_NDIM)
    x_opt = x_opt.copy()
    x_opt[2:] += np.where(x_opt[2:] >= 0, 3.0, -3.0)  # others far away
    x_opt[0] = np.eye(_NDIM)[0] * 0.5
    x_opt[1] = -np.eye(_NDIM)[0] * (0.5 + gap)
    inst = (x_opt, sigma, lambda_, M, f_max, bias)
    with jax.enable_x64(True):
        got = float(_compose(x, *inst))
    assert got == pytest.approx(_expected(x, *inst), rel=1e-12, abs=1e-9)
