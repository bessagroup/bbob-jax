"""CEC 2005 benchmark functions implemented in JAX.

IMPORTANT: This implementation replicates the CEC 2005 function FORMULAS only.
Parameters (shift vectors, rotation matrices, auxiliary matrices) are generated
from seeds rather than loaded from the official CEC 2005 data files. Results
will NOT match published CEC 2005 benchmarking results. See each function's
docstring for function-specific deviations.

Reference: Suganthan et al. (2005), "Problem Definitions and Evaluation
Criteria for the CEC 2005 Special Session on Real-Parameter Optimization."
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

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
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Shifted Schwefel 1.2 with Noise (F4).

    Noise-free version of F4 for ``jax.grad`` compatibility. The official
    CEC 2005 F4 adds Gaussian noise ``f(x) * (1 + 0.4*N(0,1))``, which is
    omitted here (``noise_omitted=True`` in
    ``cec2005_function_characteristics``).

    ![F4 3D surface](img/3d/f4.png){ width=30% }
    ![F4 2D surface](img/2d/f4.png){ width=30% }

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
    z = x - x_opt
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
    """Rastrigin without shift/offset for use as composition component."""
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0)


def _elliptic_base(z: jax.Array) -> jax.Array:
    """High conditioned elliptic without shift/rotation for composition use."""
    ndim = z.shape[-1]
    exponents = jnp.arange(ndim, dtype=jnp.float32) / jnp.maximum(ndim - 1, 1)
    coeffs = 10.0 ** (6.0 * exponents)
    return jnp.sum(coeffs * z**2)


def _sphere_base(z: jax.Array) -> jax.Array:
    return jnp.sum(z**2)


def _composition_bias() -> jax.Array:
    return jnp.array(
        [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
    )


def _height_normalize(
    fn: Callable[[jax.Array], jax.Array], ndim: int, c: float = 2000.0
) -> jax.Array:
    """Compute lambda_ for height normalization: c / |fn(5*ones(ndim))|."""
    ref = 5.0 * jnp.ones(ndim)
    val = jnp.abs(fn(ref))
    return jnp.where(val > 0.0, c / val, 1.0)


def f15(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Hybrid Composition Function 1 (F15).

    Ten Rastrigin components without rotation (identity matrices per CEC 2005
    spec). ``sigma=[1]*10``.

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
        Rotation matrix stack (overridden internally with identity matrices).
    Q : jax.Array
        Second rotation matrix stack (overridden internally with identity
        matrices).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = [_rastrigin_base] * nc
    sigma = jnp.ones(nc)
    lam_val = _height_normalize(_rastrigin_base, ndim)
    lambda_ = jnp.full(nc, lam_val)
    bias = _composition_bias()
    # F15 has no rotation: override R and Q with identity stacks
    eye_stack = jnp.stack([jnp.eye(ndim)] * nc)
    return (
        hybrid_composition(
            x, fns, sigma, lambda_, bias, x_opt, eye_stack, eye_stack
        )
        + f_opt
    )


def f16(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 1 (F16).

    Ten Rastrigin components with rotation. ``sigma=[1]*10``.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = [_rastrigin_base] * nc
    sigma = jnp.ones(nc)
    lam_val = _height_normalize(_rastrigin_base, ndim)
    lambda_ = jnp.full(nc, lam_val)
    bias = _composition_bias()
    return (
        hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R, Q) + f_opt
    )


def f17(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 1 with Noise (F17).

    Noise-free version of F17 for ``jax.grad`` compatibility. The official
    CEC 2005 F17 adds noise ``f(x) * (1 + 0.2*|N(0,1)|)``, which is omitted
    here (``noise_omitted=True`` in ``cec2005_function_characteristics``).

    ![F17 3D surface](img/3d/f17.png){ width=30% }
    ![F17 2D surface](img/2d/f17.png){ width=30% }

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    return f16(x, x_opt, f_opt, R, Q)


def _composition2_fns() -> list:
    """Return 10 component functions for Composition 2 (F18-F20)."""
    return [
        ackley,
        ackley,
        _rastrigin_base,
        _rastrigin_base,
        _elliptic_base,
        _elliptic_base,
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

    Ten mixed components (Ackley, Rastrigin, Elliptic, Weierstrass, Griewank)
    with ``sigma=[1,1,1,1,1,2,2,2,2,2]``.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = _composition2_fns()
    sigma = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    lambda_ = jnp.array([_height_normalize(fns[i], ndim) for i in range(nc)])
    bias = _composition_bias()
    return (
        hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R, Q) + f_opt
    )


def f19(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 2 with Narrow Basin (F19).

    Identical to F18 but with ``sigma[0]=0.1``, creating a narrow basin for
    the first component.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = _composition2_fns()
    sigma = jnp.array([0.1, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    lambda_ = jnp.array([_height_normalize(fns[i], ndim) for i in range(nc)])
    bias = _composition_bias()
    return (
        hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R, Q) + f_opt
    )


def f20(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Hybrid Composition 2 with Global Optimum on Bounds (F20).

    In the CEC 2005 spec, x_opt[0] is clamped to the search boundary. Here,
    x_opt is sampled from [-100, 100] as parameters are seed-generated rather
    than loaded from official data files. Formula is identical to F18.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    return f18(x, x_opt, f_opt, R, Q)


def _composition3_fns() -> list:
    """Return 10 component functions for Composition 3 (F21-F23)."""
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


def f21(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 3 (F21).

    Ten mixed components (Rastrigin, Weierstrass, Griewank, Ackley, Sphere)
    with ``sigma=[1]*10``.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = _composition3_fns()
    sigma = jnp.ones(nc)
    lambda_ = jnp.array([_height_normalize(fns[i], ndim) for i in range(nc)])
    bias = _composition_bias()
    return (
        hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R, Q) + f_opt
    )


def f22(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition 3 with High Condition Number Matrix (F22).

    In the official CEC 2005 spec, one component's rotation matrix is replaced
    with an ill-conditioned matrix. Here, standard rotation matrices are used
    and ``lambda_[0]`` is boosted by a factor of 10 to approximate the
    high-condition effect.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = _composition3_fns()
    sigma = jnp.ones(nc)
    lambda_ = jnp.array([_height_normalize(fns[i], ndim) for i in range(nc)])
    # Boost first component to approximate high-condition effect
    lambda_ = lambda_.at[0].multiply(10.0)
    bias = _composition_bias()
    return (
        hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R, Q) + f_opt
    )


def f23(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Non-Continuous Rotated Hybrid Composition Function 3 (F23).

    Continuous version of F23 for ``jax.grad`` compatibility. The official
    CEC 2005 F23 applies a rounding step
    ``y_i = round(2*x_i)/2 if |x_i - x_opt_i| >= 0.5 else x_i``, which is
    omitted here (``structure_modified=True`` in
    ``cec2005_function_characteristics``). Formula is identical to F21.

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    return f21(x, x_opt, f_opt, R, Q)


def f24(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 4 (F24).

    Same components as F21 with ``sigma=[2]*10`` (wider basins).

    ![F24 3D surface](img/3d/f24.png){ width=30% }
    ![F24 2D surface](img/2d/f24.png){ width=30% }

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    nc = 10
    fns = _composition3_fns()
    sigma = jnp.full(nc, 2.0)
    lambda_ = jnp.array([_height_normalize(fns[i], ndim) for i in range(nc)])
    bias = _composition_bias()
    return (
        hybrid_composition(x, fns, sigma, lambda_, bias, x_opt, R, Q) + f_opt
    )


def f25(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rotated Hybrid Composition Function 4 without Bounds (F25).

    Identical to F24 — neither F24 nor F25 applies a boundary penalty.

    ![F25 3D surface](img/3d/f25.png){ width=30% }
    ![F25 2D surface](img/2d/f25.png){ width=30% }

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
        Stack of second rotation matrices of shape (10, ndim, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    return f24(x, x_opt, f_opt, R, Q)
