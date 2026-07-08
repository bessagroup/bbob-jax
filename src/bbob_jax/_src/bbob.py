"""BBOB (Black-Box Optimization Benchmark) noise-free functions.

Implements the 24 BBOB benchmark functions. Each function
has the internal signature ``fn(x, x_opt, f_opt, R, Q)``
and is partially applied via the registry for end-user use.
"""

#                                                                       Modules
# =============================================================================

# Third-party
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import softjax as sj

# Local
from bbob_jax._src.utils import (
    bernoulli_vector,
    lambda_func,
    penalty,
    tasy_func,
    tosz_func,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def sphere(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere function (F1).

    Simple unimodal function with global optimum at origin.

    ![Sphere function 3D surface](img/3d/sphere.png){ width=30% }
    ![Sphere function 2D surface](img/2d/sphere.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    z = x - x_opt
    return jnp.sum(jnp.square(z)) + f_opt


def ellipsoid_seperable(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Separable ellipsoid function (F2).

    Unimodal function with high conditioning. Variables are independent.

    ![Ellipsoid seperable function 3D surface](
        img/3d/ellipsoid_seperable.png){ width=30% }
    ![Ellipsoid seperable function 2D surface](
        img/2d/ellipsoid_seperable.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    i = jnp.arange(1, ndim + 1, dtype=x.dtype)
    w = jnp.power(10.0, 6.0 * (i - 1) / (ndim - 1))
    z = tosz_func(x - x_opt)
    return jnp.sum(w * z**2) + f_opt


def rastrigin_seperable(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Separable Rastrigin function (F3).

    Highly multimodal function with many local optima.
    Variables are independent.

    ![Rastrigin seperable function 3D surface](
        img/3d/rastrigin_seperable.png){ width=30% }
    ![Rastrigin seperable function 2D surface](
        img/2d/rastrigin_seperable.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]

    alpha = lambda_func(ndim, alpha=10.0)
    temp = tosz_func(x - x_opt)
    z = jnp.matmul(alpha, tasy_func(temp, beta=0.2))

    return (
        10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z))) * jnp.sum(z**2)
        + f_opt
    )


def skew_rastrigin_bueche(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Skewed Rastrigin-Bueche function (F4).

    Multimodal function with asymmetric conditioning and skewed search space.

    ![Skew rastrigin bueche function 3D surface](
        img/3d/skew_rastrigin_bueche.png){ width=30% }
    ![Skew rastrigin bueche function 2D surface](
        img/2d/skew_rastrigin_bueche.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    i = jnp.arange(1, ndim + 1, dtype=x.dtype)
    s = jnp.power(10, 0.5 * ((i - 1) / (ndim - 1)))
    odd_indices = jnp.arange(1, ndim + 1, 2)

    z = s * tosz_func(x - x_opt)

    # Modify odd indices
    z_odd = jnp.where(z[odd_indices] > 0, z[odd_indices] * 10, z[odd_indices])
    z = z.at[odd_indices].set(z_odd)

    # Compute terms
    first_part = 10 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z)))
    second_part = jnp.sum(z * z)

    y = first_part + second_part + 100 * penalty(x)
    return y + f_opt


def linear_slope(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _ls_x_opt: jax.Array | None = None,
    _ls_s: jax.Array | None = None,
) -> jax.Array:
    """Linear slope function (F5).

    Simple linear function with a single optimum at the boundary.

    ![Linear slope function 3D surface](img/3d/linear_slope.png){ width=30% }
    ![Linear slope function 2D surface](img/2d/linear_slope.png){ width=30% }

    Note
    ----
    NaN inputs propagate to the output. The internal ``jnp.where`` that
    clamps coordinates at the boundary would otherwise mask NaN (since
    ``NaN < 25.0`` is False, selecting the finite ``x_opt`` branch), so NaN
    is re-injected element-wise to stay consistent with the other functions.

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
        Second rotation matrix.
    _ls_x_opt : jax.Array, optional
        Precomputed optimal point override (derived from Q).
    _ls_s : jax.Array, optional
        Precomputed slope vector.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]

    if _ls_x_opt is None:
        key = jr.key(0)
        key = jr.fold_in(key, Q[0, 0])
        _ls_x_opt = 5 * bernoulli_vector(ndim, key)
    if _ls_s is None:
        i = jnp.arange(1, ndim + 1, dtype=x.dtype)
        _ls_s = jnp.sign(_ls_x_opt) * jnp.power(10.0, (i - 1) / (ndim - 1))

    cond = _ls_x_opt * x < 25.0
    z = jnp.where(cond, x, _ls_x_opt)
    # `jnp.where` drops NaN inputs: `NaN < 25.0` is False, so it selects the
    # finite x_opt branch and silently clamps invalid inputs to the boundary.
    # Re-inject NaN element-wise so invalid inputs propagate, matching the
    # other 23 BBOB functions.
    z = jnp.where(jnp.isnan(x), x, z)

    result = jnp.sum(5.0 * jnp.abs(_ls_s) - _ls_s * z)
    return result + f_opt


def attractive_sector(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Attractive sector function (F6).

    Unimodal function with smooth but highly asymmetric landscape.

    ![Attractive sector function 3D surface](
        img/3d/attractive_sector.png){ width=30% }
    ![Attractive sector function 2D surface](
        img/2d/attractive_sector.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (Q @ lambda @ R).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = Q @ lambda_func(ndim, alpha=10.0) @ R
    z = _mat @ (x - x_opt)
    cond = sj.greater_st(z * x_opt, 0.0)
    s = sj.where(cond, jnp.array(100.0), jnp.array(1.0))

    term = jnp.sum((s * z) ** 2)

    result = jnp.power(tosz_func(jnp.array([term]))[0], 0.9)

    return result + f_opt


def step_ellipsoid(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Step ellipsoid function (F7).

    Unimodal function with plateau-like regions and discontinuities.

    ![Step ellipsoid function 3D surface](
        img/3d/step_ellipsoid.png){ width=30% }
    ![Step ellipsoid function 2D surface](
        img/2d/step_ellipsoid.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (lambda @ R).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    i = jnp.arange(1, ndim + 1, dtype=x.dtype)
    if _mat is None:
        _mat = lambda_func(ndim, alpha=10.0) @ R
    mult = jnp.power(10.0, 2 * ((i - 1) / (ndim - 1)))

    # Compute ẑ
    z_hat = _mat @ (x - x_opt)

    # Compute z′ using functional indexing
    z_dash = 0.5 + jnp.where(jnp.abs(z_hat) > 0.5, z_hat, 10 * z_hat)
    z_dash = jnp.floor(z_dash)
    z_dash = jnp.where(jnp.abs(z_hat) > 0.5, z_dash, z_dash / 10.0)

    # Compute z
    z = Q @ z_dash

    # Compute final f
    result = 0.1 * jnp.maximum(jnp.abs(z_hat[0]) / 1e4, jnp.sum(mult * z**2))
    return result + penalty(x) + f_opt


def rosenbrock(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock function (F8).

    Classic benchmark with narrow valley leading to the optimum.

    ![Rosenbrock function 3D surface](
        img/3d/rosenbrock.png){ width=30% }
    ![Rosenbrock function 2D surface](
        img/2d/rosenbrock.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
    # Shift and scale
    z = zmax * (x - x_opt) + 1  # shape (..., dim)

    # Create unshifted and shifted arrays along last axis
    unshift = z[..., :-1]  # all except last
    shifted = z[..., 1:]  # all except first

    # Compute the sum
    result = jnp.sum(
        100.0 * jnp.power(unshift**2 - shifted, 2)
        + jnp.power(unshift - 1.0, 2),
        axis=-1,
    )

    return result + f_opt


def rosenbrock_rotated(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock function, rotated (F9).

    Rotated version of the Rosenbrock function with increased difficulty.

    ![Rosenbrock rotated function 3D surface](
        img/3d/rosenbrock_rotated.png){ width=30% }
    ![Rosenbrock rotated function 2D surface](
        img/2d/rosenbrock_rotated.png){ width=30% }

    Note
    ----
    The true minimizer is not at ``x_opt`` but at
    ``x_opt + (0.5 / zmax) * ones @ R.T`` where
    ``zmax = max(1, sqrt(ndim) / 8)``. For some rotation matrices this
    point can fall outside the standard ``[-5, 5]`` bounds.

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)

    z = zmax * ((x - x_opt) @ R) + 0.5

    # Create unshifted and shifted arrays along last axis
    unshift = z[..., :-1]  # all except last
    shifted = z[..., 1:]  # all except first

    # Compute the sum
    result = jnp.sum(
        100.0 * jnp.power(unshift**2 - shifted, 2)
        + jnp.power(unshift - 1.0, 2),
        axis=-1,
    )

    return result + f_opt


def ellipsoid(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Ellipsoid function (F10).

    Unimodal function with high conditioning, rotated.

    ![Ellipsoid function 3D surface](
        img/3d/ellipsoid.png){ width=30% }
    ![Ellipsoid function 2D surface](
        img/2d/ellipsoid.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    idx = jnp.arange(ndim, dtype=x.dtype)
    z = tosz_func((x - x_opt) @ R)
    weights = 10.0 ** (6.0 * idx / (ndim - 1))
    return jnp.sum(weights * z**2) + f_opt


def discuss(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Discus function (F11).

    Unimodal function with one direction having much higher sensitivity.

    ![Discuss function 3D surface](img/3d/discuss.png){ width=30% }
    ![Discuss function 2D surface](img/2d/discuss.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    _ = x.shape[-1]
    z = tosz_func(R @ (x - x_opt))
    first = 1e6 * jnp.power(z[..., 0], 2)
    second = jnp.sum(jnp.power(z[..., 1:], 2), axis=-1)
    return first + second + f_opt


def bent_cigar(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Bent cigar function (F12).

    Unimodal function with a ridge, creating a cigar-like shape.

    ![Bent cigar function 3D surface](img/3d/bent_cigar.png){ width=30% }
    ![Bent cigar function 2D surface](img/2d/bent_cigar.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    _ = x.shape[-1]
    z = R @ tasy_func(R @ (x - x_opt), beta=0.5)
    return z[0] ** 2 + 1e6 * jnp.sum(z[1:] ** 2) + f_opt


def sharp_ridge(
    x: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    f_opt: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Sharp ridge function (F13).

    Unimodal function with a sharp ridge, difficult to follow.

    ![Sharp ridge function 3D surface](img/3d/sharp_ridge.png){ width=30% }
    ![Sharp ridge function 2D surface](img/2d/sharp_ridge.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix.
    f_opt : jax.Array
        Optimal function value offset.
    _mat : jax.Array, optional
        Precomputed transformation matrix (Q @ lambda @ R).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = Q @ lambda_func(ndim, alpha=10.0) @ R
    z = _mat @ (x - x_opt)
    result: jax.Array = (
        z[0] ** 2 + 100.0 * sj.sqrt(jnp.sum(z[1:] ** 2)) + f_opt
    )
    return result


def sum_of_different_powers(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sum of different powers function (F14).

    Unimodal function with different sensitivities across dimensions.

    ![Sum of different powers function 3D surface](
        img/3d/sum_of_different_powers.png){ width=30% }
    ![Sum of different powers function 2D surface](
        img/2d/sum_of_different_powers.png){ width=30% }

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    z = R @ (x - x_opt)
    idx = jnp.arange(1, ndim + 1, dtype=x.dtype)
    return jnp.sum(sj.abs_st(z) ** (2 + 4 * (idx - 1) / (ndim - 1))) + f_opt


def rastrigin(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Rastrigin function (F15).

    Highly multimodal function with many regularly distributed local
    optima.

    ![Rastrigin function 3D surface](img/3d/rastrigin.png){ width=30% }
    ![Rastrigin function 2D surface](img/2d/rastrigin.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (R @ lambda @ Q).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = R @ lambda_func(ndim, alpha=10.0) @ Q
    z = _mat @ tasy_func(tosz_func(R @ (x - x_opt)), beta=0.2)

    return (
        10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z))) * jnp.sum(z**2)
        + f_opt
    )


def weierstrass(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Weierstrass function (F16).

    Highly multimodal function with small peaks everywhere,
    continuous but non-differentiable.

    ![Weierstrass function 3D surface](img/3d/weierstrass.png){ width=30% }
    ![Weierstrass function 2D surface](img/2d/weierstrass.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (R @ lambda @ Q).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = R @ lambda_func(ndim, alpha=0.01) @ Q
    z = _mat @ tosz_func(R @ (x - x_opt))

    k = jnp.arange(0, 12, dtype=x.dtype)
    bk = 3.0**k

    f0 = jnp.sum((1 / 2**k) * jnp.cos(2 * jnp.pi * bk * 0.5))

    def inner(z: jax.Array) -> jax.Array:
        return jnp.sum(1 / 2**k * jnp.cos(2 * jnp.pi * bk * (z + 0.5))) - f0

    y = jax.vmap(inner)(z)
    sum1 = jnp.sum(y)

    first_term = 10.0 * jnp.power((1.0 / ndim) * jnp.sum(sum1), 3)
    pen = (10.0 / ndim) * penalty(x)

    return first_term + pen + f_opt


def schaffer_f7_condition_10(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Schaffer F7 function with conditioning 10 (F17).

    Multimodal function with asymmetric, moderately conditioned landscape.

    ![Schaffer f7 condition 10 function 3D surface](
        img/3d/schaffer_f7_condition_10.png){ width=30% }
    ![Schaffer f7 condition 10 function 2D surface](
        img/2d/schaffer_f7_condition_10.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (lambda @ Q).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = lambda_func(ndim, alpha=10.0) @ Q
    z = _mat @ tasy_func(R @ (x - x_opt), beta=0.5)

    s = sj.sqrt(z[:-1] ** 2 + z[1:] ** 2)

    term1 = (1 / (ndim - 1)) * jnp.sum(
        sj.sqrt(s)
        + sj.sqrt(s) * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2)
    )

    result = jnp.power(term1, 2) + 10 * penalty(x)
    return result + f_opt


def schaffer_f7_condition_1000(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Schaffer F7 function with conditioning 1000 (F18).

    Multimodal function with asymmetric, highly conditioned landscape.

    ![Schaffer f7 condition 1000 function 3D surface](
        img/3d/schaffer_f7_condition_1000.png){ width=30% }
    ![Schaffer f7 condition 1000 function 2D surface](
        img/2d/schaffer_f7_condition_1000.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (lambda @ Q).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = lambda_func(ndim, alpha=1000.0) @ Q
    z = _mat @ tasy_func(R @ (x - x_opt), beta=0.5)

    s = sj.sqrt(z[:-1] ** 2 + z[1:] ** 2)

    term1 = (1 / (ndim - 1)) * jnp.sum(
        sj.sqrt(s)
        + sj.sqrt(s) * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2)
    )

    result = jnp.power(term1, 2) + 10 * penalty(x)
    return result + f_opt


def griewank_rosenbrock_f8f2(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Griewank-Rosenbrock F8F2 function (F19).

    Multimodal function combining Rosenbrock's narrow valley with
    Griewank's modulation.

    ![Griewank rosenbrock f8f2 function 3D surface](
        img/3d/griewank_rosenbrock_f8f2.png){ width=30% }
    ![Griewank rosenbrock f8f2 function 2D surface](
        img/2d/griewank_rosenbrock_f8f2.png){ width=30% }

    Note
    ----
    The true minimizer is not at ``x_opt`` but at
    ``x_opt + (0.5 / zmax) * R.T @ ones`` where
    ``zmax = max(1, sqrt(ndim) / 8)``. For some rotation matrices this
    point can fall outside the standard ``[-5, 5]`` bounds.

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
        Second rotation matrix.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    z = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0) * (R @ (x - x_opt)) + 0.5
    s = 100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1) ** 2

    return (10 / (ndim - 1)) * jnp.sum((s / 4000) - jnp.cos(s)) + 10.0 + f_opt


def schwefel_xsinx(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _sw_ones: jax.Array | None = None,
    _sw_x_opt_shape: jax.Array | None = None,
    _sw_lamb: jax.Array | None = None,
    _sw_f_ref: jax.Array | None = None,
) -> jax.Array:
    """Schwefel x*sin(x) function (F20).

    Multimodal function with many local optima and a global optimum
    far from origin.

    ![Schwefel xsinx function 3D surface](
        img/3d/schwefel_xsinx.png){ width=30% }
    ![Schwefel xsinx function 2D surface](
        img/2d/schwefel_xsinx.png){ width=30% }

    Note
    ----
    The zero-offset constant (≈4.1898) is computed dynamically from the
    internal ``x_opt_shape`` rather than hardcoded as a precision-specific
    literal. This ensures exact cancellation at the optimum at whichever
    precision JAX is configured for (float32 or float64), avoiding the
    ~5e-7 residual a hardcoded float64 literal would leave behind under
    x32, and the equivalent precision mismatch under x64.

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
        Second rotation matrix.
    _sw_ones : jax.Array, optional
        Precomputed Bernoulli sign vector (derived from Q).
    _sw_x_opt_shape : jax.Array, optional
        Precomputed optimal shape vector.
    _sw_lamb : jax.Array, optional
        Precomputed conditioning matrix (lambda).
    _sw_f_ref : jax.Array, optional
        Precomputed reference function value for zero-offset.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]

    if _sw_ones is None:
        key = jr.key(0)
        key = jr.fold_in(key, Q[0, 0])
        _sw_ones = bernoulli_vector(ndim, key)
    if _sw_x_opt_shape is None:
        _sw_x_opt_shape = 4.2096874633 / 2 * _sw_ones
    if _sw_lamb is None:
        _sw_lamb = lambda_func(ndim, alpha=10.0)

    # helper for shift
    x_trans = x - x_opt + _sw_x_opt_shape
    x_hat = 2.0 * _sw_ones * x_trans

    z_hat = x_hat.at[..., 1:].add(
        0.25 * (x_hat[..., :-1] - 2.0 * jnp.abs(_sw_x_opt_shape[..., :-1]))
    )

    z = 100.0 * (
        _sw_lamb @ (z_hat - 2.0 * jnp.abs(_sw_x_opt_shape))
        + 2.0 * jnp.abs(_sw_x_opt_shape)
    )

    f = (
        -1.0
        / (100.0 * ndim)
        * jnp.sum(z * jnp.sin(sj.sqrt(jnp.abs(z))), axis=-1)
    )

    # Penalization
    pen = 100.0 * penalty(z / 100.0)

    if _sw_f_ref is None:
        z_ref = 200.0 * jnp.abs(_sw_x_opt_shape)
        _sw_f_ref = (
            1.0 / (100.0 * ndim) * jnp.sum(z_ref * jnp.sin(jnp.sqrt(z_ref)))
        )
    return f + _sw_f_ref + pen + f_opt


def _precompute_gallagher(
    x_opt: jax.Array,
    Q: jax.Array,
    ndim: int,
    num_peaks: int,
    w_divisor: int,
    alpha_first: float,
    y_minval: float,
    y_maxval: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precompute Gallagher peak weights and locations.

    Parameters
    ----------
    x_opt : jax.Array
        Optimal point for the first peak.
    Q : jax.Array
        Rotation matrix (used to seed the RNG).
    ndim : int
        Number of input dimensions.
    num_peaks : int
        Number of Gallagher peaks.
    w_divisor : int
        Divisor for computing peak weights.
    alpha_first : float
        Conditioning number for the first peak.
    y_minval : float
        Lower bound for random peak locations.
    y_maxval : float
        Upper bound for random peak locations.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Peak weights, peak locations, and conditioning
        diagonal vectors.
    """
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

    # Compute diagonal vectors instead of full (ndim x ndim) matrices.
    # lambda_func(ndim, alpha_i) = diag(alpha_i^(idx / (2*(ndim-1))))
    # Then divided by alpha_i^0.25.
    idx = jnp.arange(ndim, dtype=float)
    # alpha: (num_peaks,), idx: (ndim,) -> c_diags: (num_peaks, ndim)
    c_diags = jnp.power(
        alpha[:, None], idx[None, :] / (2 * (ndim - 1))
    ) / jnp.power(alpha[:, None], 0.25)

    return w, y, c_diags


def gallagher_101_peaks(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None = None,
    _gal_y: jax.Array | None = None,
    _gal_c_diags: jax.Array | None = None,
) -> jax.Array:
    """Gallagher 101 peaks function (F21).

    Multimodal function with 101 optima of different heights.

    ![Gallagher 101 peaks function 3D surface](
        img/3d/gallagher_101_peaks.png){ width=30% }
    ![Gallagher 101 peaks function 2D surface](
        img/2d/gallagher_101_peaks.png){ width=30% }

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
        Second rotation matrix.
    _gal_w : jax.Array, optional
        Precomputed peak weight vector of shape (101,).
    _gal_y : jax.Array, optional
        Precomputed peak location matrix of shape (101, ndim).
    _gal_c_diags : jax.Array, optional
        Precomputed conditioning diagonal vectors of shape (101, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]

    if _gal_w is None:
        w, y, c_diags = _precompute_gallagher(
            x_opt, Q, ndim, 101, 99, 1000.0, -5.0, 5.0
        )
    else:
        assert _gal_y is not None
        assert _gal_c_diags is not None
        w, y, c_diags = _gal_w, _gal_y, _gal_c_diags

    diff = x[None, :] - y  # (101, ndim)
    rotated_diff = jnp.einsum("ij,...j->...i", R, diff)  # (101, ndim)
    exponents = -(1.0 / (2.0 * ndim)) * jnp.sum(
        c_diags * rotated_diff**2, axis=-1
    )  # (101,)
    inside_max = w * jnp.exp(exponents)  # (101,)

    f = 10.0 - sj.max_st(inside_max, axis=0)

    f_tosz = tosz_func(jnp.array([f]))[0]

    result = jnp.power(f_tosz, 2) + penalty(x)

    return result + f_opt


def gallagher_21_peaks(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None = None,
    _gal_y: jax.Array | None = None,
    _gal_c_diags: jax.Array | None = None,
) -> jax.Array:
    """Gallagher 21 peaks function (F22).

    Multimodal function with 21 optima of different heights.

    ![Gallagher 21 peaks function 3D surface](
        img/3d/gallagher_21_peaks.png){ width=30% }
    ![Gallagher 21 peaks function 2D surface](
        img/2d/gallagher_21_peaks.png){ width=30% }

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
        Second rotation matrix.
    _gal_w : jax.Array, optional
        Precomputed peak weight vector of shape (21,).
    _gal_y : jax.Array, optional
        Precomputed peak location matrix of shape (21, ndim).
    _gal_c_diags : jax.Array, optional
        Precomputed conditioning diagonal vectors of shape (21, ndim).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]

    if _gal_w is None:
        w, y, c_diags = _precompute_gallagher(
            x_opt, Q, ndim, 21, 19, 1000.0**2, -4.9, 4.9
        )
    else:
        assert _gal_y is not None
        assert _gal_c_diags is not None
        w, y, c_diags = _gal_w, _gal_y, _gal_c_diags

    diff = x[None, :] - y  # (21, ndim)
    rotated_diff = jnp.einsum("ij,...j->...i", R, diff)  # (21, ndim)
    exponents = -(1.0 / (2.0 * ndim)) * jnp.sum(
        c_diags * rotated_diff**2, axis=-1
    )  # (21,)
    inside_max = w * jnp.exp(exponents)  # (21,)

    f = 10.0 - sj.max_st(inside_max, axis=0)

    f_tosz = tosz_func(jnp.array([f]))[0]

    result = jnp.power(f_tosz, 2) + penalty(x)

    return result + f_opt


def katsuura(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Katsuura function (F23).

    Highly multimodal function with many small local optima,
    rugged landscape.

    ![Katsuura function 3D surface](img/3d/katsuura.png){ width=30% }
    ![Katsuura function 2D surface](img/2d/katsuura.png){ width=30% }

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
        Second rotation matrix.
    _mat : jax.Array, optional
        Precomputed transformation matrix (Q @ lambda @ R).

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    if _mat is None:
        _mat = Q @ lambda_func(ndim, alpha=100.0) @ R
    z = _mat @ (x - x_opt)

    J = 2.0 ** jnp.arange(1, 33, dtype=float)  # (32,)
    # jsum term: shape (32, dim)
    z_expanded = z[None, :]  # (1, dim)
    J_expanded = J[:, None]  # (32, 1)
    jsum = (
        jnp.abs(J_expanded * z_expanded - jnp.round(J_expanded * z_expanded))
        / J_expanded
    )  # (32, dim)

    # Sum over j (the 32 terms)
    sum_j = jnp.sum(jsum, axis=0)  # (dim,)

    # Multiply by (1..dim) and add 1
    bracket = 1.0 + jnp.arange(1, ndim + 1, dtype=x.dtype) * sum_j  # (dim,)
    prod = jnp.prod(bracket)

    # Final scaling and power
    prod = jnp.power(prod, 10.0 / ndim**1.2)
    prod = prod * (10.0 / ndim**2.0) - (10.0 / ndim**2.0)

    return cast(jax.Array, prod + penalty(x) + f_opt)


def lunacek_bi_rastrigin(
    x: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    f_opt: jax.Array,
    _mat: jax.Array | None = None,
    _x_opt_shape: jax.Array | None = None,
    _s: jax.Array | None = None,
    _mu1: jax.Array | None = None,
) -> jax.Array:
    """Lunacek bi-Rastrigin function (F24).

    Highly multimodal function with two funnels and many local optima.

    ![Lunacek bi rastrigin function 3D surface](
        img/3d/lunacek_bi_rastrigin.png){ width=30% }
    ![Lunacek bi rastrigin function 2D surface](
        img/2d/lunacek_bi_rastrigin.png){ width=30% }

    Parameters
    ----------
    x : jax.Array
        Input array of shape (..., ndim).
    x_opt : jax.Array
        Optimal point.
    R : jax.Array
        Rotation matrix.
    Q : jax.Array
        Second rotation matrix.
    f_opt : jax.Array
        Optimal function value offset.
    _mat : jax.Array, optional
        Precomputed transformation matrix (Q @ lambda @ R).
    _x_opt_shape : jax.Array, optional
        Precomputed optimal shape vector (derived from Q).
    _s : jax.Array, optional
        Precomputed conditioning parameter.
    _mu1 : jax.Array, optional
        Precomputed second funnel center.

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    mu0 = 2.5
    d = 1.0

    if _x_opt_shape is None:
        key = jr.key(0)
        key = jr.fold_in(key, Q[0, 0])
        _x_opt_shape = (mu0 / 2.0) * bernoulli_vector(ndim, key)
    if _s is None:
        _s = 1.0 - 1.0 / (2.0 * jnp.sqrt(ndim + 20.0) - 8.2)
    if _mu1 is None:
        _mu1 = -jnp.sqrt((mu0**2 - d) / _s)
    if _mat is None:
        _mat = Q @ lambda_func(ndim, alpha=100.0) @ R

    # Shift x so that x_opt corresponds to x_opt_shape in the transformed space
    x_trans = x - x_opt + _x_opt_shape
    x_hat = 2 * jnp.sign(_x_opt_shape) * x_trans

    z = _mat @ (x_hat - mu0 * jnp.ones_like(x))

    term1 = sj.min_st(
        jnp.stack(
            [
                jnp.sum(jnp.power(x_hat - mu0, 2)),
                d * ndim + _s * jnp.sum(jnp.power(x_hat - _mu1, 2)),
            ]
        ),
    )

    term2 = 10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z)))

    result: jax.Array = term1 + term2 + 1e4 * penalty(x) + f_opt
    return result
