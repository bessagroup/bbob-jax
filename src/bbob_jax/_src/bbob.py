#                                                                       Modules
# =============================================================================

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr

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
    return jnp.sum(jnp.square(z))


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
) -> jax.Array:
    """Linear slope function (F5).

    Simple linear function with a single optimum at the boundary.

    ![Linear slope function 3D surface](img/3d/linear_slope.png){ width=30% }
    ![Linear slope function 2D surface](img/2d/linear_slope.png){ width=30% }

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
    key = jr.key(0)
    key = jr.fold_in(key, Q[0, 0])

    x_opt = 5 * bernoulli_vector(ndim, key)
    i = jnp.arange(1, ndim + 1, dtype=x.dtype)
    s = jnp.sign(x_opt) * jnp.power(10.0, (i - 1) / (ndim - 1))

    cond = x_opt * x < 25.0
    z = jnp.where(cond, x, x_opt)

    result = jnp.sum(5.0 * jnp.abs(s) - s * z)
    return result + f_opt


def attractive_sector(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=10.0)
    z = Q @ lamb @ R @ (x - x_opt)
    cond = (z * x_opt) > 0.0
    s = jnp.where(cond, 100.0, 1.0)

    term = jnp.sum((s * z) ** 2)

    result = jnp.power(tosz_func(jnp.array([term]))[0], 0.9)

    return result + f_opt


def step_ellipsoid(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    i = jnp.arange(1, ndim + 1, dtype=x.dtype)
    lamb = lambda_func(ndim, alpha=10.0)
    mult = jnp.power(10.0, 2 * ((i - 1) / (ndim - 1)))

    # Compute ẑ
    z_hat = lamb @ R @ (x - x_opt)

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

    z = zmax * (x @ R) + 0.5

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
    z = tosz_func(x @ R)
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=10.0)
    z = Q @ lamb @ R @ (x - x_opt)
    return z[0] ** 2 + 100.0 * jnp.sqrt(jnp.sum(z[1:] ** 2)) + f_opt


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
    return jnp.sum(jnp.abs(z) ** (2 + 4 * (idx - 1) / (ndim - 1))) + f_opt


def rastrigin(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=10.0)
    z = R @ lamb @ Q @ tasy_func(tosz_func(R @ (x - x_opt)), beta=0.2)

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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=0.01)
    z = R @ lamb @ Q @ tosz_func(R @ (x - x_opt))

    k = jnp.arange(0, 12, dtype=x.dtype)
    bk = 3.0**k

    f0 = jnp.sum((1 / 2**k) * jnp.cos(2 * jnp.pi * bk * 0.5))

    def inner(z):
        return 1 / 2**k * jnp.cos(2 * jnp.pi * bk * (z + 0.5)) - f0

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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=10.0)
    z = lamb @ Q @ tasy_func(R @ (x - x_opt), beta=0.5)

    s = jnp.sqrt(z[:-1] ** 2 + z[1:] ** 2)

    term1 = (1 / (ndim - 1)) * jnp.sum(
        jnp.sqrt(s)
        + jnp.sqrt(s) * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2)
    )

    result = jnp.power(term1, 2) + 10 * penalty(x)
    return result + f_opt


def schaffer_f7_condition_1000(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=1000.0)
    z = lamb @ Q @ tasy_func(R @ (x - x_opt), beta=0.5)

    s = jnp.sqrt(z[:-1] ** 2 + z[1:] ** 2)

    term1 = (1 / (ndim - 1)) * jnp.sum(
        jnp.sqrt(s)
        + jnp.sqrt(s) * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2)
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
    z = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0) * (R @ x) + 0.5
    s = 100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1) ** 2

    return (10 / (ndim - 1)) * jnp.sum((s / 4000) - jnp.cos(s)) + 10.0 + f_opt


def schwefel_xsinx(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Schwefel x*sin(x) function (F20).

    Multimodal function with many local optima and a global optimum
    far from origin.

    ![Schwefel xsinx function 3D surface](
        img/3d/schwefel_xsinx.png){ width=30% }
    ![Schwefel xsinx function 2D surface](
        img/2d/schwefel_xsinx.png){ width=30% }

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
    key = jr.key(0)
    key = jr.fold_in(key, Q[0, 0])
    ones = bernoulli_vector(ndim, key)
    lamb = lambda_func(ndim, alpha=10.0)
    x_hat = 2.0 * ones * x
    x_opt = 4.2096874633 / 2 * ones

    z_hat = x_hat.at[..., 1:].add(
        0.25 * (x_hat[..., :-1] - 2.0 * jnp.abs(x_opt[..., :-1]))
    )

    z = 100.0 * (lamb @ (z_hat - 2.0 * jnp.abs(x_opt)) + 2.0 * jnp.abs(x_opt))

    f = (
        -1.0
        / (100.0 * ndim)
        * jnp.sum(z * jnp.sin(jnp.sqrt(jnp.abs(z))), axis=-1)
    )

    # Penalization
    pen = 100.0 * penalty(z / 100.0)

    return f + 4.189828872724339 + pen + f_opt


def gallagher_101_peaks(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    key = jr.key(0)
    key = jr.fold_in(key, Q[0, 0])
    key1, key2 = jr.split(key)
    i = jnp.arange(1, 102, dtype=jnp.float32)
    j = jnp.arange(0, 100, dtype=jnp.float32)

    w = 1.1 + 8.0 * ((i - 2) / 99)
    w = w.at[0].set(10.0)

    a = jnp.power(1000, 2.0 * (j / 99.0))
    alpha = jr.permutation(key1, a)
    alpha = jnp.concatenate([jnp.array([1000.0]), alpha])

    y = jr.uniform(key2, shape=(*i.shape, ndim), minval=-5.0, maxval=5.0)
    x_opt = jr.uniform(key, shape=(ndim,), minval=-4.0, maxval=4.0)
    y = y.at[0].set(x_opt)

    C = jax.vmap(lambda_func, in_axes=(None, 0))(ndim, alpha)

    C /= jnp.power(alpha, 0.25)[:, None, None]

    diff = x[None, :] - y

    def apply_C(C_i, d_i, w_i):
        val = -(1.0 / (2.0 * ndim)) * d_i.T @ (R.T @ C_i @ R) @ d_i
        return w_i * jnp.exp(val)

    inside_max = jax.vmap(apply_C)(C, diff, w)

    f = 10.0 - jnp.max(inside_max, axis=0)

    f_tosz = tosz_func(jnp.array([f]))[0]

    result = jnp.power(f_tosz, 2) + penalty(x)

    return result + f_opt


def gallagher_21_peaks(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    key = jr.key(0)
    key = jr.fold_in(key, Q[0, 0])
    key1, key2 = jr.split(key)
    i = jnp.arange(1, 22, dtype=jnp.float32)
    j = jnp.arange(0, 20, dtype=jnp.float32)

    w = 1.1 + 8.0 * ((i - 2) / 19)
    w = w.at[0].set(10.0)

    a = jnp.power(1000, 2.0 * (j / 19.0))
    alpha = jr.permutation(key1, a)
    alpha = jnp.concatenate([jnp.array([1000.0**2]), alpha])

    y = jr.uniform(key2, shape=(*i.shape, ndim), minval=-4.9, maxval=4.9)
    x_opt = jr.uniform(key, shape=(ndim,), minval=-3.92, maxval=3.92)
    y = y.at[0].set(x_opt)

    C = jax.vmap(lambda_func, in_axes=(None, 0))(ndim, alpha)

    C /= jnp.power(alpha, 0.25)[:, None, None]

    diff = x[None, :] - y + f_opt

    def apply_C(C_i, d_i, w_i):
        val = -(1.0 / (2.0 * ndim)) * d_i.T @ (R.T @ C_i @ R) @ d_i
        return w_i * jnp.exp(val)

    inside_max = jax.vmap(apply_C)(C, diff, w)

    f = 10.0 - jnp.max(inside_max, axis=0)

    f_tosz = tosz_func(jnp.array([f]))[0]

    result = jnp.power(f_tosz, 2) + penalty(x)

    return result + f_opt


def katsuura(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=100.0)
    z = Q @ lamb @ R @ (x - x_opt)

    J = 2.0 ** jnp.arange(1, 33, dtype=jnp.float32)  # (32,)
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

    return prod + penalty(x) + f_opt


def lunacek_bi_rastrigin(
    x: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    f_opt: jax.Array,
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

    Returns
    -------
    jax.Array
        Function value(s).
    """
    ndim = x.shape[-1]
    key = jr.key(0)
    key = jr.fold_in(key, Q[0, 0])
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * jnp.sqrt(ndim + 20.0) - 8.2)

    x_opt = (mu0 / 2.0) * bernoulli_vector(ndim, key)
    x_hat = 2 * jnp.sign(x_opt) * x

    lamb = lambda_func(ndim, alpha=100.0)
    z = Q @ lamb @ R @ (x_hat - mu0 * jnp.ones_like(x))

    mu1 = -jnp.sqrt((mu0**2 - d) / s)

    term1 = jnp.minimum(
        jnp.sum(jnp.power(x_hat - mu0, 2)),
        d * ndim + s * jnp.sum(jnp.power(x_hat - mu1, 2)),
    )

    term2 = 10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z)))

    return term1 + term2 + 1e4 * penalty(x) + f_opt
