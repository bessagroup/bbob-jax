"""BBOB-noisy suite implementations (f101-f130).

Thirty stochastic benchmark functions: eight base landscapes,
each disturbed by the Gaussian, uniform or Cauchy noise model
from ``noise.py`` at moderate (f101-f106) or severe (f107-f130)
severity. Replicates the legacy COCO reference
(``benchmarksnoisy.c``); the deterministic (undisturbed) path is
cross-checked against the compiled C code.

Every function follows the assembly of the reference: the noise
model disturbs only the *residual* (the base value above the
optimum); the boundary penalty — always ``100 * penalty(x)`` in
this suite — and ``f_opt`` are added outside the noise, to both
the disturbed and the undisturbed value.

The undisturbed value of each base is exposed as a ``*_true``
function with the same bound-parameter signature minus ``key``;
:func:`bbob_jax._src.problem.problem` binds it as
``Problem.fn_true``.

The shared ``transforms.py`` ``tosz_func``/``tasy_func`` are
reference-exact (since ADR 0005) and are reused here; the C
cross-check of this suite doubles as their verification.
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import cast

# Third-party
import jax
import jax.numpy as jnp
import softjax as sj

# Local
from bbob_jax._src.bbob import _precompute_gallagher
from bbob_jax._src.noise import cauchy_noise, gauss_noise, uniform_noise
from bbob_jax._src.transforms import (
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


#                                                                Base residuals
# =============================================================================
# The undisturbed base value above the optimum — no boundary penalty, no
# f_opt. This is the quantity the noise models disturb.


def _sphere_residual(x: jax.Array, x_opt: jax.Array) -> jax.Array:
    """Sphere: ``sum((x - x_opt)^2)``."""
    return jnp.sum(jnp.square(x - x_opt))


def _rosenbrock_residual(x: jax.Array, x_opt: jax.Array) -> jax.Array:
    """Non-rotated Rosenbrock with dimension-dependent scaling.

    ``z = max(1, sqrt(D)/8) * (x - x_opt) + 1``; the bound
    ``x_opt`` is the minimizer (the reference samples a shift and
    scales it by 0.75; here the minimizer is sampled directly,
    matching the noiseless suite's ``rosenbrock``).
    """
    ndim = x.shape[-1]
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
    z = zmax * (x - x_opt) + 1.0
    return jnp.sum(
        100.0 * jnp.power(z[:-1] ** 2 - z[1:], 2) + jnp.power(z[:-1] - 1.0, 2)
    )


def _step_ellipsoid_residual(
    x: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None,
) -> jax.Array:
    """Step ellipsoid, condition 100, rounding resolution 10."""
    ndim = x.shape[-1]
    if _mat is None:
        _mat = lambda_func(ndim, alpha=10.0) @ R
    i = jnp.arange(ndim, dtype=x.dtype)
    mult = jnp.power(100.0, i / max(ndim - 1, 1))

    z_hat = _mat @ (x - x_opt)

    z_dash = 0.5 + jnp.where(jnp.abs(z_hat) > 0.5, z_hat, 10 * z_hat)
    z_dash = jnp.floor(z_dash)
    z_dash = jnp.where(jnp.abs(z_hat) > 0.5, z_dash, z_dash / 10.0)

    z = Q @ z_dash
    return 0.1 * jnp.maximum(jnp.abs(z_hat[0]) * 1e-4, jnp.sum(mult * z**2))


def _ellipsoid_residual(
    x: jax.Array, x_opt: jax.Array, R: jax.Array
) -> jax.Array:
    """Rotated ellipsoid, condition 1e4, legacy T_osz."""
    ndim = x.shape[-1]
    i = jnp.arange(ndim, dtype=x.dtype)
    weights = jnp.power(1e4, i / max(ndim - 1, 1))
    z = tosz_func(R @ (x - x_opt))
    return jnp.sum(weights * z**2)


def _different_powers_residual(
    x: jax.Array, x_opt: jax.Array, R: jax.Array
) -> jax.Array:
    """Sum of different powers between x^2 and x^6, with sqrt."""
    ndim = x.shape[-1]
    z = R @ (x - x_opt)
    i = jnp.arange(ndim, dtype=x.dtype)
    exponents = 2.0 + 4.0 * i / max(ndim - 1, 1)
    return cast(jax.Array, sj.sqrt(jnp.sum(sj.abs_st(z) ** exponents)))


def _schaffer_f7_residual(
    x: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None,
) -> jax.Array:
    """Schaffer F7, condition 10, legacy asymmetry beta=0.5."""
    ndim = x.shape[-1]
    if _mat is None:
        _mat = lambda_func(ndim, alpha=10.0) @ Q
    z = _mat @ tasy_func(R @ (x - x_opt), beta=0.5)

    s = sj.sqrt(z[:-1] ** 2 + z[1:] ** 2)
    term = jnp.sum(
        sj.sqrt(s)
        + sj.sqrt(s) * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2)
    )
    return jnp.power(term / max(ndim - 1, 1), 2)


def _griewank_rosenbrock_residual(
    x: jax.Array, x_opt: jax.Array, R: jax.Array
) -> jax.Array:
    """Griewank-Rosenbrock F8F2 blocks, noisy-suite scaling.

    ``1 + sum(s/4000 - cos(s)) / (D - 1)`` — a factor 10 smaller
    than the noiseless suite's ``10 + 10 * ... `` variant. The
    reference has no shift; the minimizer is
    ``x_opt + (0.5 / zmax) * R.T @ ones`` (see the spec resolver).
    """
    ndim = x.shape[-1]
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
    z = zmax * (R @ (x - x_opt)) + 0.5
    s = 100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2
    return jnp.sum(s / 4000.0 - jnp.cos(s)) / max(ndim - 1, 1) + 1.0


def _gallagher_residual(
    x: jax.Array,
    x_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None,
    _gal_y_rot: jax.Array | None,
    _gal_c_diags: jax.Array | None,
) -> jax.Array:
    """Gallagher 101 Gaussian peaks, condition up to 1000."""
    ndim = x.shape[-1]
    if _gal_w is None:
        w, y_rot, c_diags = _precompute_gallagher(
            x_opt, R, Q, ndim, 101, 99, 1000.0, -5.0, 5.0
        )
    else:
        assert _gal_y_rot is not None
        assert _gal_c_diags is not None
        w, y_rot, c_diags = _gal_w, _gal_y_rot, _gal_c_diags

    rotated_diff = R @ x - y_rot  # (101, ndim)
    exponents = -(1.0 / (2.0 * ndim)) * jnp.sum(
        c_diags * rotated_diff**2, axis=-1
    )
    f = 10.0 - jnp.max(w * jnp.exp(exponents), axis=0)
    return jnp.power(tosz_func(f), 2)


#                                                          Undisturbed variants
# =============================================================================
# Same bound-parameter signature as the noisy functions minus ``key``;
# bound as ``Problem.fn_true`` with identical instance parameters.


def sphere_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Undisturbed sphere (base of f101-f103, f107-f109).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    return _sphere_residual(x, x_opt) + 100.0 * penalty(x) + f_opt


def rosenbrock_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Undisturbed Rosenbrock (base of f104-f106, f110-f112).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    return _rosenbrock_residual(x, x_opt) + 100.0 * penalty(x) + f_opt


def step_ellipsoid_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Undisturbed step ellipsoid (base of f113-f115).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    residual = _step_ellipsoid_residual(x, x_opt, R, Q, _mat)
    return residual + 100.0 * penalty(x) + f_opt


def ellipsoid_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Undisturbed ellipsoid, condition 1e4 (base of f116-f118).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    return _ellipsoid_residual(x, x_opt, R) + 100.0 * penalty(x) + f_opt


def different_powers_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Undisturbed sum of different powers (base of f119-f121).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    residual = _different_powers_residual(x, x_opt, R)
    return residual + 100.0 * penalty(x) + f_opt


def schaffer_f7_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Undisturbed Schaffer F7, condition 10 (base of f122-f124).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    residual = _schaffer_f7_residual(x, x_opt, R, Q, _mat)
    return residual + 100.0 * penalty(x) + f_opt


def griewank_rosenbrock_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Undisturbed Griewank-Rosenbrock (base of f125-f127).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
        Undisturbed function value.
    """
    residual = _griewank_rosenbrock_residual(x, x_opt, R)
    return residual + 100.0 * penalty(x) + f_opt


def gallagher_true(
    x: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None = None,
    _gal_y_rot: jax.Array | None = None,
    _gal_c_diags: jax.Array | None = None,
) -> jax.Array:
    """Undisturbed Gallagher 101 peaks (base of f128-f130).

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
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
    _gal_y_rot : jax.Array, optional
        Precomputed rotated peak locations ``y @ R.T``,
        shape (101, ndim).
    _gal_c_diags : jax.Array, optional
        Precomputed conditioning diagonals, shape (101, ndim).

    Returns
    -------
    jax.Array
        Undisturbed function value.
    """
    residual = _gallagher_residual(
        x, x_opt, R, Q, _gal_w, _gal_y_rot, _gal_c_diags
    )
    return residual + 100.0 * penalty(x) + f_opt


#                                                     Moderate noise: f101-f106
# =============================================================================


def f101(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere with moderate Gaussian noise (f101).

    ``gauss_noise(sphere, beta=0.01)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _sphere_residual(x, x_opt)
    disturbed = gauss_noise(residual, key, beta=0.01)
    return disturbed + 100.0 * penalty(x) + f_opt


def f102(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere with moderate uniform noise (f102).

    ``uniform_noise(sphere, alpha=0.01*(0.49+1/D), beta=0.01)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _sphere_residual(x, x_opt)
    disturbed = uniform_noise(
        residual, key, alpha=0.01 * (0.49 + 1.0 / ndim), beta=0.01
    )
    return disturbed + 100.0 * penalty(x) + f_opt


def f103(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere with moderate seldom Cauchy noise (f103).

    ``cauchy_noise(sphere, alpha=0.01, p=0.05)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _sphere_residual(x, x_opt)
    disturbed = cauchy_noise(residual, key, alpha=0.01, p=0.05)
    return disturbed + 100.0 * penalty(x) + f_opt


def f104(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock with moderate Gaussian noise (f104).

    ``gauss_noise(rosenbrock, beta=0.01)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _rosenbrock_residual(x, x_opt)
    disturbed = gauss_noise(residual, key, beta=0.01)
    return disturbed + 100.0 * penalty(x) + f_opt


def f105(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock with moderate uniform noise (f105).

    ``uniform_noise(rosenbrock, alpha=0.01*(0.49+1/D), beta=0.01)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _rosenbrock_residual(x, x_opt)
    disturbed = uniform_noise(
        residual, key, alpha=0.01 * (0.49 + 1.0 / ndim), beta=0.01
    )
    return disturbed + 100.0 * penalty(x) + f_opt


def f106(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock with moderate seldom Cauchy noise (f106).

    ``cauchy_noise(rosenbrock, alpha=0.01, p=0.05)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _rosenbrock_residual(x, x_opt)
    disturbed = cauchy_noise(residual, key, alpha=0.01, p=0.05)
    return disturbed + 100.0 * penalty(x) + f_opt


#                                                       Severe noise: f107-f130
# =============================================================================


def f107(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere with Gaussian noise (f107).

    ``gauss_noise(sphere, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _sphere_residual(x, x_opt)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f108(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere with uniform noise (f108).

    ``uniform_noise(sphere, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _sphere_residual(x, x_opt)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f109(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sphere with seldom Cauchy noise (f109).

    ``cauchy_noise(sphere, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _sphere_residual(x, x_opt)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f110(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock with Gaussian noise (f110).

    ``gauss_noise(rosenbrock, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _rosenbrock_residual(x, x_opt)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f111(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock with uniform noise (f111).

    ``uniform_noise(rosenbrock, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _rosenbrock_residual(x, x_opt)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f112(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Rosenbrock with seldom Cauchy noise (f112).

    ``cauchy_noise(rosenbrock, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _rosenbrock_residual(x, x_opt)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f113(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Step ellipsoid with Gaussian noise (f113).

    ``gauss_noise(step_ellipsoid, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _step_ellipsoid_residual(x, x_opt, R, Q, _mat)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f114(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Step ellipsoid with uniform noise (f114).

    ``uniform_noise(step_ellipsoid, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _step_ellipsoid_residual(x, x_opt, R, Q, _mat)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f115(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Step ellipsoid with seldom Cauchy noise (f115).

    ``cauchy_noise(step_ellipsoid, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _step_ellipsoid_residual(x, x_opt, R, Q, _mat)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f116(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Ellipsoid (condition 1e4) with Gaussian noise (f116).

    ``gauss_noise(ellipsoid, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _ellipsoid_residual(x, x_opt, R)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f117(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Ellipsoid (condition 1e4) with uniform noise (f117).

    ``uniform_noise(ellipsoid, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _ellipsoid_residual(x, x_opt, R)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f118(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Ellipsoid (condition 1e4) with seldom Cauchy noise (f118).

    ``cauchy_noise(ellipsoid, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _ellipsoid_residual(x, x_opt, R)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f119(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sum of different powers with Gaussian noise (f119).

    ``gauss_noise(different_powers, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _different_powers_residual(x, x_opt, R)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f120(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sum of different powers with uniform noise (f120).

    ``uniform_noise(different_powers, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _different_powers_residual(x, x_opt, R)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f121(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Sum of different powers with seldom Cauchy noise (f121).

    ``cauchy_noise(different_powers, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _different_powers_residual(x, x_opt, R)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f122(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Schaffer F7 with Gaussian noise (f122).

    ``gauss_noise(schaffer_f7, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _schaffer_f7_residual(x, x_opt, R, Q, _mat)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f123(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Schaffer F7 with uniform noise (f123).

    ``uniform_noise(schaffer_f7, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _schaffer_f7_residual(x, x_opt, R, Q, _mat)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f124(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _mat: jax.Array | None = None,
) -> jax.Array:
    """Schaffer F7 with seldom Cauchy noise (f124).

    ``cauchy_noise(schaffer_f7, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _schaffer_f7_residual(x, x_opt, R, Q, _mat)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f125(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Griewank-Rosenbrock F8F2 with Gaussian noise (f125).

    ``gauss_noise(griewank_rosenbrock, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _griewank_rosenbrock_residual(x, x_opt, R)
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f126(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Griewank-Rosenbrock F8F2 with uniform noise (f126).

    ``uniform_noise(griewank_rosenbrock, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _griewank_rosenbrock_residual(x, x_opt, R)
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f127(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> jax.Array:
    """Griewank-Rosenbrock F8F2 with seldom Cauchy noise (f127).

    ``cauchy_noise(griewank_rosenbrock, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
        Disturbed function value.
    """
    residual = _griewank_rosenbrock_residual(x, x_opt, R)
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt


def f128(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None = None,
    _gal_y_rot: jax.Array | None = None,
    _gal_c_diags: jax.Array | None = None,
) -> jax.Array:
    """Gallagher 101 peaks with Gaussian noise (f128).

    ``gauss_noise(gallagher, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
    _gal_y_rot : jax.Array, optional
        Precomputed rotated peak locations ``y @ R.T``,
        shape (101, ndim).
    _gal_c_diags : jax.Array, optional
        Precomputed conditioning diagonals, shape (101, ndim).

    Returns
    -------
    jax.Array
        Disturbed function value.
    """
    residual = _gallagher_residual(
        x, x_opt, R, Q, _gal_w, _gal_y_rot, _gal_c_diags
    )
    disturbed = gauss_noise(residual, key, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f129(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None = None,
    _gal_y_rot: jax.Array | None = None,
    _gal_c_diags: jax.Array | None = None,
) -> jax.Array:
    """Gallagher 101 peaks with uniform noise (f129).

    ``uniform_noise(gallagher, alpha=0.49+1/D, beta=1)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
    _gal_y_rot : jax.Array, optional
        Precomputed rotated peak locations ``y @ R.T``,
        shape (101, ndim).
    _gal_c_diags : jax.Array, optional
        Precomputed conditioning diagonals, shape (101, ndim).

    Returns
    -------
    jax.Array
        Disturbed function value.
    """
    ndim = x.shape[-1]
    residual = _gallagher_residual(
        x, x_opt, R, Q, _gal_w, _gal_y_rot, _gal_c_diags
    )
    disturbed = uniform_noise(residual, key, alpha=0.49 + 1.0 / ndim, beta=1.0)
    return disturbed + 100.0 * penalty(x) + f_opt


def f130(
    x: jax.Array,
    key: jax.Array,
    x_opt: jax.Array,
    f_opt: jax.Array,
    R: jax.Array,
    Q: jax.Array,
    _gal_w: jax.Array | None = None,
    _gal_y_rot: jax.Array | None = None,
    _gal_c_diags: jax.Array | None = None,
) -> jax.Array:
    """Gallagher 101 peaks with seldom Cauchy noise (f130).

    ``cauchy_noise(gallagher, alpha=1, p=0.2)``.

    Parameters
    ----------
    x : jax.Array
        Input array of shape (ndim,).
    key : jax.Array
        JAX PRNGKey consumed by the noise model.
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
    _gal_y_rot : jax.Array, optional
        Precomputed rotated peak locations ``y @ R.T``,
        shape (101, ndim).
    _gal_c_diags : jax.Array, optional
        Precomputed conditioning diagonals, shape (101, ndim).

    Returns
    -------
    jax.Array
        Disturbed function value.
    """
    residual = _gallagher_residual(
        x, x_opt, R, Q, _gal_w, _gal_y_rot, _gal_c_diags
    )
    disturbed = cauchy_noise(residual, key, alpha=1.0, p=0.2)
    return disturbed + 100.0 * penalty(x) + f_opt
