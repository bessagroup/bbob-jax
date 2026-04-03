"""CEC 2005 benchmark functions implemented in JAX.

IMPORTANT: This implementation targets CEC 2005 formula parity where that can
coexist with ``jax.grad`` compatibility. Parameters (shift vectors, rotation
matrices, auxiliary matrices) are generated from seeds rather than loaded from
the official CEC 2005 data files. A few non-continuous or winner-take-all
operations are replaced with smooth approximations so all public functions
remain differentiable enough for JAX autodiff use. Functions with stochastic
noise (F4, F17, F24, F25) require a JAX PRNGKey as a second argument.
Results will NOT match published CEC 2005 benchmarking results.
See each function's docstring and ``cec2005_function_characteristics`` for
function-specific deviations.

Reference: Suganthan et al. (2005), "Problem Definitions and Evaluation
Criteria for the CEC 2005 Special Session on Real-Parameter Optimization."
"""

from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import softjax as sj

from bbob_jax._src.utils import (
    ackley,
    cec2005_weierstrass,
    griewank,
    hybrid_composition,
    scaffer_f6,
)

__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"


def f1(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Sphere function (F1).

    Simple unimodal function with global optimum at x_opt.

    ![F1 3D surface](img/3d/f1.png){ width=30% }
    ![F1 2D surface](img/2d/f1.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt
    return jnp.sum(z**2) + f_opt


def f2(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Schwefel's Problem 1.2 (F2).

    Unimodal function with non-separable variables via cumulative sum.

    ![F2 3D surface](img/3d/f2.png){ width=30% }
    ![F2 2D surface](img/2d/f2.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt
    return jnp.sum(jnp.cumsum(z) ** 2) + f_opt


def f3(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rotated High Conditioned Elliptic function (F3).

    Unimodal function with high conditioning and rotation applied.

    ![F3 3D surface](img/3d/f3.png){ width=30% }
    ![F3 2D surface](img/2d/f3.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    z = R @ (x - x_opt)
    exponents = jnp.arange(ndim, dtype=jnp.float32) / jnp.maximum(ndim - 1, 1)
    coeffs = 10.0 ** (6.0 * exponents)
    return jnp.sum(coeffs * z**2) + f_opt


def f4(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Schwefel 1.2 with Noise (F4).

    Applies multiplicative Gaussian noise: ``f(x) * (1 + 0.4 * N(0,1))``.

    ![F4 3D surface](img/3d/f4.png){ width=30% }
    ![F4 2D surface](img/2d/f4.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    key : jax.Array
        JAX PRNGKey for stochastic noise.
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt
    base = jnp.sum(jnp.cumsum(z) ** 2)
    return base * (1 + 0.4 * jr.normal(key, shape=())) + f_opt


def f5(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Schwefel's Problem 2.6 with Global Optimum on Bounds (F5).

    Non-differentiable function based on the Chebyshev norm. R is repurposed
    as the A matrix (n x n). In the official CEC 2005 spec, x_opt has
    components clamped to ±5; here x_opt is sampled from [-100, 100] as
    parameters are seed-generated rather than loaded from official data files.

    ![F5 3D surface](img/3d/f5.png){ width=30% }
    ![F5 2D surface](img/2d/f5.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Repurposed as the A matrix (n x n); b = A @ x_opt is computed
        internally.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    diff = R @ (x - x_opt)
    return jnp.max(jnp.abs(diff)) + f_opt


def f6(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rosenbrock's function (F6).

    Multimodal function with a narrow curved valley. Input is shifted by
    ``x - x_opt + 1`` to place the valley at x_opt.

    ![F6 3D surface](img/3d/f6.png){ width=30% }
    ![F6 2D surface](img/2d/f6.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt + 1.0
    return (
        jnp.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2)
        + f_opt
    )


def f7(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rotated Griewank's without Bounds (F7).

    Multimodal function with many regularly distributed local optima.

    ![F7 3D surface](img/3d/f7.png){ width=30% }
    ![F7 2D surface](img/2d/f7.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return griewank(z) + f_opt


def f8(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rotated Ackley's with Global Optimum on Bounds (F8).

    Multimodal function with many local optima and a nearly flat outer region.

    ![F8 3D surface](img/3d/f8.png){ width=30% }
    ![F8 2D surface](img/2d/f8.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return ackley(z) + f_opt


def f9(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rastrigin's function (F9).

    Highly multimodal function with many local optima arranged in a grid.
    Variables are not rotated.

    ![F9 3D surface](img/3d/f9.png){ width=30% }
    ![F9 2D surface](img/2d/f9.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0) + f_opt


def f10(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rotated Rastrigin's function (F10).

    Highly multimodal function with many local optima and rotation applied.

    ![F10 3D surface](img/3d/f10.png){ width=30% }
    ![F10 2D surface](img/2d/f10.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0) + f_opt


def f11(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rotated Weierstrass function (F11).

    Continuous but differentiable only finitely many times. Highly multimodal.

    ![F11 3D surface](img/3d/f11.png){ width=30% }
    ![F11 2D surface](img/2d/f11.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    return cec2005_weierstrass(z) + f_opt


def f12(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Schwefel's Problem 2.13 (F12).

    Unimodal function defined via inner-product matrices. R is repurposed as
    the 'a' matrix (n x n), Q as the 'b' matrix (n x n), and x_opt stores the
    alpha vector (optimal solution).

    ![F12 3D surface](img/3d/f12.png){ width=30% }
    ![F12 2D surface](img/2d/f12.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Alpha vector (optimal solution angles).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Repurposed as the 'a' matrix (n x n).
    Q : jax.Array
        Repurposed as the 'b' matrix (n x n).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    A = jnp.sum(
        R * jnp.sin(x_opt)[None, :] + Q * jnp.cos(x_opt)[None, :], axis=-1
    )
    B = jnp.sum(R * jnp.sin(x)[None, :] + Q * jnp.cos(x)[None, :], axis=-1)
    return jnp.sum((A - B) ** 2) + f_opt


def f13(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Expanded Extended Griewank's plus Rosenbrock's F8F2 (F13).

    Applies a 1D Griewank function to consecutive Rosenbrock values cyclically.
    R and Q are unused.

    ![F13 3D surface](img/3d/f13.png){ width=30% }
    ![F13 2D surface](img/2d/f13.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix (unused).
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt + 1.0
    zi = z[:-1]
    zi1 = z[1:]
    rosen_vals = 100.0 * (zi**2 - zi1) ** 2 + (zi - 1.0) ** 2
    rosen_last = 100.0 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1.0) ** 2
    rosen_all = jnp.concatenate([rosen_vals, jnp.array([rosen_last])])
    # 1D Griewank: g(y) = y^2/4000 - cos(y) + 1
    g_vals = rosen_all**2 / 4000.0 - jnp.cos(rosen_all) + 1.0
    return jnp.sum(g_vals) + f_opt


def f14(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Rotated Expanded Scaffer's F6 (F14).

    Applies Scaffer's F6 to consecutive pairs cyclically after rotation.

    ![F14 3D surface](img/3d/f14.png){ width=30% }
    ![F14 2D surface](img/2d/f14.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix (unused).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = R @ (x - x_opt)
    pairs_val = jax.vmap(scaffer_f6)(z[:-1], z[1:])
    last_val = scaffer_f6(z[-1], z[0])
    return jnp.sum(pairs_val) + last_val + f_opt


def _rastrigin_base(z: jax.Array) -> jax.Array:
    """Rastrigin without shift/offset for composition use."""
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0)


def _elliptic_base(z: jax.Array) -> jax.Array:
    """High conditioned elliptic for composition use."""
    ndim = z.shape[-1]
    exponents = jnp.arange(ndim, dtype=jnp.float32) / jnp.maximum(ndim - 1, 1)
    coeffs = 10.0 ** (6.0 * exponents)
    return jnp.sum(coeffs * z**2)


def _sphere_base(z: jax.Array) -> jax.Array:
    return jnp.sum(z**2)


def _expanded_scaffer_f6_base(z: jax.Array) -> jax.Array:
    """Expanded Scaffer's F6 applied cyclically."""
    pairs = jax.vmap(scaffer_f6)(z[:-1], z[1:])
    last = scaffer_f6(z[-1], z[0])
    return jnp.sum(pairs) + last


def _f8f2_base(z: jax.Array) -> jax.Array:
    """F8F2: Griewank composed with Rosenbrock, applied cyclically.

    F2(x,y) = 100*(x^2 - y)^2 + (x - 1)^2
    F8(t) = t^2/4000 - cos(t) + 1
    Result = sum of F8(F2(z_i, z_{i+1})) cyclically.
    """
    z1 = z[:-1]
    z2 = z[1:]
    f2_vals = 100.0 * (z1**2 - z2) ** 2 + (z1 - 1.0) ** 2
    f2_last = 100.0 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1.0) ** 2
    f2_all = jnp.concatenate([f2_vals, jnp.array([f2_last])])
    f8_vals = f2_all**2 / 4000.0 - jnp.cos(f2_all) + 1.0
    return jnp.sum(f8_vals)


def _soft_round_input(z: jax.Array) -> jax.Array:
    """Apply soft rounding: y_j = round(2*x_j)/2 if |x_j|>=0.5."""
    diff = sj.abs(z) - 0.5
    mask = sj.heaviside(diff)
    z_rounded = sj.round(2.0 * z) / 2.0
    return cast(jax.Array, mask * z_rounded + (1.0 - mask) * z)


def _non_continuous_scaffer_f6_base(z: jax.Array) -> jax.Array:
    """Non-continuous Expanded Scaffer's F6 with soft rounding."""
    y = _soft_round_input(z)
    return _expanded_scaffer_f6_base(y)


def _non_continuous_rastrigin_base(z: jax.Array) -> jax.Array:
    """Non-continuous Rastrigin with soft rounding."""
    y = _soft_round_input(z)
    return _rastrigin_base(y)


def _sphere_noisy_base(z: jax.Array, key: jax.Array) -> jax.Array:
    """Sphere with multiplicative absolute Gaussian noise."""
    return jnp.sum(z**2) * (1 + 0.01 * jnp.abs(jr.normal(key, shape=())))


def _composition_bias() -> jax.Array:
    return jnp.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
        ]
    )


# --- Component function lists per composition group ---


def _composition1_fns() -> list:
    """F15/F16/F17: Rast, Weier, Griew, Ackley, Sphere."""
    return [
        _rastrigin_base,
        _rastrigin_base,
        cec2005_weierstrass,
        cec2005_weierstrass,
        griewank,
        griewank,
        ackley,
        ackley,
        _sphere_base,
        _sphere_base,
    ]


def f15(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Hybrid Composition Function 1 (F15).

    Ten mixed components (Rastrigin, Weierstrass, Griewank, Ackley,
    Sphere) without rotation (identity matrices). ``sigma=[1]*10``.

    ![F15 3D surface](img/3d/f15.png){ width=30% }
    ![F15 2D surface](img/2d/f15.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Rotation matrix stack (overridden with identity).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    fns = _composition1_fns()
    sigma = jnp.ones(10)
    lambda_ = jnp.array(
        [
            1,
            1,
            10,
            10,
            5 / 60,
            5 / 60,
            5 / 32,
            5 / 32,
            5 / 100,
            5 / 100,
        ]
    )
    bias = _composition_bias()
    eye = jnp.stack([jnp.eye(ndim)] * 10)
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, eye) + f_opt


def f16(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 1 (F16).

    Same as F15 but with rotation matrices (condition number 2).

    ![F16 3D surface](img/3d/f16.png){ width=30% }
    ![F16 2D surface](img/2d/f16.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    fns = _composition1_fns()
    sigma = jnp.ones(10)
    lambda_ = jnp.array(
        [
            1,
            1,
            10,
            10,
            5 / 60,
            5 / 60,
            5 / 32,
            5 / 32,
            5 / 100,
            5 / 100,
        ]
    )
    bias = _composition_bias()
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R) + f_opt


def f17(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 1 with Noise (F17).

    Applies multiplicative absolute Gaussian noise to the base F16 result:
    ``f(x) * (1 + 0.4 * |N(0,1)|)``.

    ![F17 3D surface](img/3d/f17.png){ width=30% }
    ![F17 2D surface](img/2d/f17.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    key : jax.Array
        JAX PRNGKey for stochastic noise.
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    base = f16(x, x_opt, f_opt, R, Q) - f_opt
    return base * (1 + 0.4 * jnp.abs(jr.normal(key, shape=()))) + f_opt


def _composition2_fns() -> list:
    """F18/F19/F20: Ackley, Rast, Sphere, Weier, Griew."""
    return [
        ackley,
        ackley,
        _rastrigin_base,
        _rastrigin_base,
        _sphere_base,
        _sphere_base,
        cec2005_weierstrass,
        cec2005_weierstrass,
        griewank,
        griewank,
    ]


def f18(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 2 (F18).

    Ten mixed components (Ackley, Rastrigin, Sphere, Weierstrass,
    Griewank). ``o10 = [0,...,0]`` sets a local optimum at origin.

    ![F18 3D surface](img/3d/f18.png){ width=30% }
    ![F18 2D surface](img/2d/f18.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    fns = _composition2_fns()
    sigma = jnp.array(
        [
            1.0,
            2.0,
            1.5,
            1.5,
            1.0,
            1.0,
            1.5,
            1.5,
            2.0,
            2.0,
        ]
    )
    lambda_ = jnp.array(
        [
            2 * 5 / 32,
            5 / 32,
            2 * 1,
            1,
            2 * 5 / 100,
            5 / 100,
            2 * 10,
            10,
            2 * 5 / 60,
            5 / 60,
        ]
    )
    bias = _composition_bias()
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R) + f_opt


def f19(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 2, Narrow Basin (F19).

    Same as F18 but with ``sigma[0]=0.1`` and ``lambda[0]``
    scaled by 0.1.

    ![F19 3D surface](img/3d/f19.png){ width=30% }
    ![F19 2D surface](img/2d/f19.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    fns = _composition2_fns()
    sigma = jnp.array(
        [
            0.1,
            2.0,
            1.5,
            1.5,
            1.0,
            1.0,
            1.5,
            1.5,
            2.0,
            2.0,
        ]
    )
    lambda_ = jnp.array(
        [
            0.1 * 5 / 32,
            5 / 32,
            2 * 1,
            1,
            2 * 5 / 100,
            5 / 100,
            2 * 10,
            10,
            2 * 5 / 60,
            5 / 60,
        ]
    )
    bias = _composition_bias()
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R) + f_opt


def f20(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Hybrid Composition 2 with Global Optimum on Bounds (F20).

    Same as F18. x_opt[0] has even-indexed dims (1-based) set to 5
    by the registry factory.

    ![F20 3D surface](img/3d/f20.png){ width=30% }
    ![F20 2D surface](img/2d/f20.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    return f18(x, x_opt, f_opt, R, Q)


def _composition3_fns() -> list:
    """F21/F22/F23: Scaffer, Rast, F8F2, Weier, Griew."""
    return [
        _expanded_scaffer_f6_base,
        _expanded_scaffer_f6_base,
        _rastrigin_base,
        _rastrigin_base,
        _f8f2_base,
        _f8f2_base,
        cec2005_weierstrass,
        cec2005_weierstrass,
        griewank,
        griewank,
    ]


def _composition4_fns() -> list:
    """F24/F25: 10 different components."""
    return [
        cec2005_weierstrass,
        _expanded_scaffer_f6_base,
        _f8f2_base,
        ackley,
        _rastrigin_base,
        griewank,
        _non_continuous_scaffer_f6_base,
        _non_continuous_rastrigin_base,
        _elliptic_base,
        _sphere_noisy_base,
    ]


def f21(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 3 (F21).

    Ten mixed components (Scaffer's F6, Rastrigin, F8F2,
    Weierstrass, Griewank).

    ![F21 3D surface](img/3d/f21.png){ width=30% }
    ![F21 2D surface](img/2d/f21.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    fns = _composition3_fns()
    sigma = jnp.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
    lambda_ = jnp.array(
        [
            5 * 5 / 100,
            5 / 100,
            5 * 1,
            1,
            5 * 1,
            1,
            5 * 10,
            10,
            5 * 5 / 200,
            5 / 200,
        ]
    )
    bias = _composition_bias()
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R) + f_opt


def f22(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition 3, High Condition Number (F22).

    Same as F21 but the factory supplies high-condition-number matrices
    [10, 20, 50, 100, 200, 1000, 2000, 3000, 4000, 5000].

    ![F22 3D surface](img/3d/f22.png){ width=30% }
    ![F22 2D surface](img/2d/f22.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    fns = _composition3_fns()
    sigma = jnp.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
    lambda_ = jnp.array(
        [
            5 * 5 / 100,
            5 / 100,
            5 * 1,
            1,
            5 * 1,
            1,
            5 * 10,
            10,
            5 * 5 / 200,
            5 / 200,
        ]
    )
    bias = _composition_bias()
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R) + f_opt


def f23(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Non-Continuous Rotated Hybrid Composition Function 3 (F23).

    Same as F21 but with soft rounding applied to x before
    evaluation. Uses ``softjax`` for ``jax.grad`` compatibility.

    ![F23 3D surface](img/3d/f23.png){ width=30% }
    ![F23 2D surface](img/2d/f23.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    # Soft rounding: y_j = round(2*x_j)/2 if |x_j - o1_j| >= 0.5
    o = x_opt[0]
    diff = sj.abs(x - o) - 0.5
    mask = sj.heaviside(diff)
    x_rounded = sj.round(2.0 * x) / 2.0
    y = mask * x_rounded + (1.0 - mask) * x
    return f21(y, x_opt, f_opt, R, Q)


def f24(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 4 (F24).

    Ten different components including non-continuous variants.
    Rotation matrices with condition numbers
    [100, 50, 30, 10, 5, 5, 4, 3, 2, 2] are supplied by the factory.
    The 10th component (noisy sphere) applies multiplicative absolute
    Gaussian noise: ``sphere(z) * (1 + 0.01 * |N(0,1)|)``.

    ![F24 3D surface](img/3d/f24.png){ width=30% }
    ![F24 2D surface](img/2d/f24.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    key : jax.Array
        JAX PRNGKey for stochastic noise in the noisy sphere component.
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    fns = _composition4_fns()
    fns[-1] = lambda z: _sphere_noisy_base(z, key)
    sigma = jnp.full(10, 2.0)
    lambda_ = jnp.array(
        [
            10,
            5 / 20,
            1,
            5 / 32,
            1,
            5 / 100,
            5 / 50,
            1,
            5 / 100,
            5 / 100,
        ]
    )
    bias = _composition_bias()
    return hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R) + f_opt


def f25(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 4 without Bounds (F25).

    Identical to F24. In the paper, this function differs only by the
    initialization range. The implementation keeps the same function formula
    while not constraining component optima to the initialization interval.

    ![F25 3D surface](img/3d/f25.png){ width=30% }
    ![F25 2D surface](img/2d/f25.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    key : jax.Array
        JAX PRNGKey for stochastic noise (forwarded to F24).
    x_opt : jax.Array
        Matrix of component optima of shape (10, ndim).
    f_opt : jax.Array
        Optimal function value offset.
    R : jax.Array
        Stack of rotation matrices of shape (10, ndim, ndim).
    Q : jax.Array
        Unused.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    return f24(x, key, x_opt, f_opt, R, Q)
