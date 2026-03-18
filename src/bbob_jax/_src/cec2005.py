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
    """F1: Shifted Sphere. Minimum f_opt at x_opt."""
    z = x - x_opt
    return jnp.sum(z**2) + f_opt


def f2(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F2: Shifted Schwefel's Problem 1.2. Minimum f_opt at x_opt."""
    z = x - x_opt
    return jnp.sum(jnp.cumsum(z) ** 2) + f_opt


def f3(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F3: Shifted Rotated High Conditioned Elliptic. Minimum f_opt at x_opt.
    R is the rotation matrix. Q is unused."""
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
    """F4: Shifted Schwefel 1.2 (noise omitted for jax.grad compatibility).

    The official CEC 2005 F4 adds Gaussian noise: f(x) * (1 + 0.4*N(0,1)).
    Noise is omitted here. noise_omitted=True in
    cec2005_function_characteristics.
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
    """F5: Schwefel's Problem 2.6 with Global Optimum on Bounds.

    R is repurposed as the A matrix (n x n). x_opt is the global optimum
    (±5 per dim in the original; here sampled from [-100, 100] — parameters
    are seed-generated, not from official CEC 2005 data files).

    f(x) = max_i(|sum_j(A_ij * x_j) - b_i|) where b = A @ x_opt
          = max(|R @ (x - x_opt)|)
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
    """F6: Shifted Rosenbrock's. Minimum f_opt at x_opt.
    Shift is x - x_opt + 1 to place valley at x_opt."""
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
    """F7: Shifted Rotated Griewank's without Bounds. Minimum f_opt at
    x_opt."""
    from bbob_jax._src.utils import griewank

    z = R @ (x - x_opt)
    return griewank(z) + f_opt


def f8(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F8: Shifted Rotated Ackley's with Global Optimum on Bounds.
    R is the rotation matrix; Q is unused."""
    from bbob_jax._src.utils import ackley

    z = R @ (x - x_opt)
    return ackley(z) + f_opt


def f9(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F9: Shifted Rastrigin's. Minimum f_opt at x_opt."""
    z = x - x_opt
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0) + f_opt


def f10(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F10: Shifted Rotated Rastrigin's. Minimum f_opt at x_opt."""
    z = R @ (x - x_opt)
    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0) + f_opt


def f11(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F11: Shifted Rotated Weierstrass. Minimum f_opt at x_opt."""
    from bbob_jax._src.utils import cec2005_weierstrass

    z = R @ (x - x_opt)
    return cec2005_weierstrass(z) + f_opt


def f12(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """F12: Schwefel's Problem 2.13.

    R is repurposed as the 'a' matrix (n x n), Q as the 'b' matrix (n x n).
    x_opt stores the alpha vector (optimal solution).

    A_i = sum_j(a_ij * sin(alpha_j) + b_ij * cos(alpha_j))
    B_i(x) = sum_j(a_ij * sin(x_j) + b_ij * cos(x_j))
    f(x) = sum_i((A_i - B_i(x))^2) + f_opt
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
    """F13: Expanded Extended Griewank's plus Rosenbrock's F8F2.

    Applies g(rosenbrock(x_i, x_{i+1})) cyclically, where g is the 1D Griewank.
    x_opt shifts the input. R and Q are unused.
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
    """F14: Shifted Rotated Expanded Scaffer's F6.

    Applies Scaffer F6 to consecutive pairs (z_i, z_{i+1}) cyclically.
    """
    from bbob_jax._src.utils import scaffer_f6

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
    """F15: Hybrid Composition Function 1 (10 Rastrigin, no rotation).
    sigma=[1]*10. F15 uses identity rotation per component per CEC 2005 spec.
    """
    from bbob_jax._src.utils import hybrid_composition

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
    """F16: Rotated Hybrid Composition Function 1 (10 Rastrigin).
    sigma=[1]*10. Uses R and Q rotation matrices from factory.
    """
    from bbob_jax._src.utils import hybrid_composition

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
    """F17: Rotated Hybrid Composition Function 1 with Noise (noise omitted).

    The official CEC 2005 F17 adds noise: f(x) * (1 + 0.2*|N(0,1)|).
    Noise is omitted here for jax.grad compatibility.
    noise_omitted=True in cec2005_function_characteristics.
    """
    return f16(x, x_opt, f_opt, R, Q)


def f18(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f19(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f20(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f21(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f22(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f23(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f24(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError


def f25(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    raise NotImplementedError
