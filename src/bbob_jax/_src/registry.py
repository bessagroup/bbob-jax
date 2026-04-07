"""Registry factories for BBOB and CEC 2005 benchmark functions.

Provides ``registry`` and ``registry_original`` dicts that map
function names to factories. Each factory partially applies
random or deterministic parameters (x_opt, f_opt, R, Q) to
produce a callable benchmark function.
"""

#                                                                       Modules
# =============================================================================

# Standard
import math
from collections.abc import Callable
from typing import Any, cast

# Third-Party
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray

from bbob_jax._src.bbob import (
    _precompute_gallagher,
    attractive_sector,
    bent_cigar,
    discuss,
    ellipsoid,
    ellipsoid_seperable,
    gallagher_21_peaks,
    gallagher_101_peaks,
    griewank_rosenbrock_f8f2,
    katsuura,
    linear_slope,
    lunacek_bi_rastrigin,
    rastrigin,
    rastrigin_seperable,
    rosenbrock,
    rosenbrock_rotated,
    schaffer_f7_condition_10,
    schaffer_f7_condition_1000,
    schwefel_xsinx,
    sharp_ridge,
    skew_rastrigin_bueche,
    sphere,
    step_ellipsoid,
    sum_of_different_powers,
    weierstrass,
)
from bbob_jax._src.cec2005 import (
    _composition1_fns,
    _composition2_fns,
    _composition3_fns,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f25,
)
from bbob_jax._src.utils import (
    bernoulli_vector,
    compute_composition_f_max,
    fopt,
    lambda_func,
    rotation_matrix,
    xopt,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BBOBFn = tuple[Callable[[jax.Array], jax.Array], jax.Array]


def _partial_keywords(
    fn_partial: Callable[[jax.Array], jax.Array],
) -> dict[str, Any]:
    """Extract keyword arguments from a Partial."""
    return cast(dict[str, Any], cast(Any, fn_partial).keywords)


def _conditioned_linear_transform(
    dim: int, key: PRNGKeyArray, condition_number: float
) -> jax.Array:
    """Generate a seeded CEC-style linear transform.

    Parameters
    ----------
    dim : int
        Matrix dimension.
    key : PRNGKeyArray
        JAX random key.
    condition_number : float
        Target condition number.

    Returns
    -------
    jax.Array
        Transform matrix of shape ``(dim, dim)``.
    """
    key_p, key_q, key_u = jr.split(key, 3)
    p = rotation_matrix(dim, key_p)
    q = rotation_matrix(dim, key_q)
    u = jr.uniform(key_u, shape=(dim,), minval=0.0, maxval=1.0)
    span = jnp.maximum(jnp.max(u) - jnp.min(u), 1e-12)
    exponents = (u - jnp.min(u)) / span
    n = jnp.diag(jnp.asarray(condition_number, dtype=jnp.float32) ** exponents)
    return p @ n @ q


def _conditioned_transform_stack(
    dim: int, keys: PRNGKeyArray, condition_numbers: jax.Array
) -> jax.Array:
    """Stack conditioned linear transforms for each component."""
    return jnp.stack(
        [
            _conditioned_linear_transform(
                dim, keys[i], float(condition_numbers[i])
            )
            for i in range(len(condition_numbers))
        ]
    )


def _sample_nonsingular_integer_matrix(
    key: PRNGKeyArray, ndim: int, minval: int, maxval: int
) -> jax.Array:
    """Sample an integer matrix, retrying if singular.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    ndim : int
        Matrix dimension.
    minval : int
        Minimum integer value (inclusive).
    maxval : int
        Maximum integer value (inclusive).

    Returns
    -------
    jax.Array
        Non-singular integer matrix of shape
        ``(ndim, ndim)``.
    """
    attempt_keys = jr.split(key, 32)
    fallback = jr.randint(
        attempt_keys[0], (ndim, ndim), minval, maxval + 1
    ).astype(jnp.float32)
    for attempt_key in attempt_keys:
        mat = jr.randint(attempt_key, (ndim, ndim), minval, maxval + 1).astype(
            jnp.float32
        )
        if not jnp.isclose(jnp.linalg.det(mat), 0.0):
            return mat
    return fallback


def make_determinstic(
    fn: Callable, ndim: int, key: PRNGKeyArray | None = None
) -> BBOBFn:
    """Deterministic BBOB factory with zero shift and identity rotations.

    Parameters
    ----------
    fn : Callable
        Base BBOB function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        Ignored; accepted for signature compatibility.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    x_opt = jnp.zeros(ndim)
    eye = jnp.eye(ndim)
    f_opt = jnp.array(0.0)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=eye, Q=eye), f_opt


def make_randomized(fn: Callable, ndim: int, key: PRNGKeyArray) -> BBOBFn:
    """Randomized BBOB factory with random shift and rotations.

    Parameters
    ----------
    fn : Callable
        Base BBOB function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray
        JAX random key for parameter generation.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    key1, key2 = jr.split(key)
    x_opt = xopt(key=key1, ndim=ndim, minval=-4.0, maxval=4.0)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    f_opt = fopt(key)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=R, Q=Q), f_opt


def _make_with_mat(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    alpha: float = 10.0,
    order: str = "QLR",
) -> BBOBFn:
    """Factory that precomputes a matrix chain.

    Combines ``lambda_func`` conditioning with rotation
    matrices in the specified ``order``.
    """
    partial_fn, f_opt_val = make_randomized(fn, ndim, key)
    kw = _partial_keywords(partial_fn)
    lamb = lambda_func(ndim, alpha)
    if order == "QLR":
        mat = kw["Q"] @ lamb @ kw["R"]
    elif order == "RLQ":
        mat = kw["R"] @ lamb @ kw["Q"]
    elif order == "LQ":
        mat = lamb @ kw["Q"]
    elif order == "LR":
        mat = lamb @ kw["R"]
    else:
        raise ValueError(f"Unknown order: {order}")
    return Partial(fn, **kw, _mat=mat), f_opt_val


def _make_with_mat_deterministic(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    alpha: float = 10.0,
    order: str = "QLR",
) -> BBOBFn:
    """Deterministic factory that precomputes a matrix chain.

    Combines ``lambda_func`` conditioning with identity
    rotations in the specified ``order``.
    """
    partial_fn, f_opt_val = make_determinstic(fn, ndim)
    kw = _partial_keywords(partial_fn)
    lamb = lambda_func(ndim, alpha)
    if order == "QLR":
        mat = kw["Q"] @ lamb @ kw["R"]
    elif order == "RLQ":
        mat = kw["R"] @ lamb @ kw["Q"]
    elif order == "LQ":
        mat = lamb @ kw["Q"]
    elif order == "LR":
        mat = lamb @ kw["R"]
    else:
        raise ValueError(f"Unknown order: {order}")
    return Partial(fn, **kw, _mat=mat), f_opt_val


def _make_linear_slope(fn: Callable, ndim: int, key: PRNGKeyArray) -> BBOBFn:
    """Factory for linear_slope with precomputed random state."""
    partial_fn, f_opt_val = make_randomized(fn, ndim, key)
    kw = _partial_keywords(partial_fn)
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    ls_x_opt = 5 * bernoulli_vector(ndim, rng)
    i = jnp.arange(1, ndim + 1, dtype=jnp.float32)
    ls_s = jnp.sign(ls_x_opt) * jnp.power(10.0, (i - 1) / (ndim - 1))
    return Partial(fn, **kw, _ls_x_opt=ls_x_opt, _ls_s=ls_s), f_opt_val


def _make_linear_slope_deterministic(
    fn: Callable, ndim: int, key: PRNGKeyArray | None = None
) -> BBOBFn:
    """Deterministic factory for linear_slope."""
    partial_fn, f_opt_val = make_determinstic(fn, ndim)
    kw = _partial_keywords(partial_fn)
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    ls_x_opt = 5 * bernoulli_vector(ndim, rng)
    i = jnp.arange(1, ndim + 1, dtype=jnp.float32)
    ls_s = jnp.sign(ls_x_opt) * jnp.power(10.0, (i - 1) / (ndim - 1))
    return Partial(fn, **kw, _ls_x_opt=ls_x_opt, _ls_s=ls_s), f_opt_val


def _make_schwefel(fn: Callable, ndim: int, key: PRNGKeyArray) -> BBOBFn:
    """Factory for schwefel_xsinx with precomputed random state + lambda."""
    partial_fn, f_opt_val = make_randomized(fn, ndim, key)
    kw = _partial_keywords(partial_fn)
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    ones = bernoulli_vector(ndim, rng)
    x_opt_shape = 4.2096874633 / 2 * ones
    lamb = lambda_func(ndim, alpha=10.0)
    z_ref = 200.0 * jnp.abs(x_opt_shape)
    f_ref = 1.0 / (100.0 * ndim) * jnp.sum(z_ref * jnp.sin(jnp.sqrt(z_ref)))
    return Partial(
        fn,
        **kw,
        _sw_ones=ones,
        _sw_x_opt_shape=x_opt_shape,
        _sw_lamb=lamb,
        _sw_f_ref=f_ref,
    ), f_opt_val


def _make_schwefel_deterministic(
    fn: Callable, ndim: int, key: PRNGKeyArray | None = None
) -> BBOBFn:
    """Deterministic factory for schwefel_xsinx."""
    partial_fn, f_opt_val = make_determinstic(fn, ndim)
    kw = _partial_keywords(partial_fn)
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    ones = bernoulli_vector(ndim, rng)
    x_opt_shape = 4.2096874633 / 2 * ones
    lamb = lambda_func(ndim, alpha=10.0)
    z_ref = 200.0 * jnp.abs(x_opt_shape)
    f_ref = 1.0 / (100.0 * ndim) * jnp.sum(z_ref * jnp.sin(jnp.sqrt(z_ref)))
    return Partial(
        fn,
        **kw,
        _sw_ones=ones,
        _sw_x_opt_shape=x_opt_shape,
        _sw_lamb=lamb,
        _sw_f_ref=f_ref,
    ), f_opt_val


def _make_lunacek_randomized(
    fn: Callable, ndim: int, key: PRNGKeyArray
) -> BBOBFn:
    """Factory for lunacek_bi_rastrigin with precomputed state."""
    partial_fn, f_opt_val = make_randomized(fn, ndim, key)
    kw = _partial_keywords(partial_fn)
    mat = kw["Q"] @ lambda_func(ndim, alpha=100.0) @ kw["R"]
    # Precompute random-state-dependent values
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    mu0 = 2.5
    d = 1.0
    x_opt_shape = (mu0 / 2.0) * bernoulli_vector(ndim, rng)
    s = 1.0 - 1.0 / (2.0 * jnp.sqrt(ndim + 20.0) - 8.2)
    mu1 = -jnp.sqrt((mu0**2 - d) / s)
    return Partial(
        fn,
        **kw,
        _mat=mat,
        _x_opt_shape=x_opt_shape,
        _s=s,
        _mu1=mu1,
    ), f_opt_val


def _make_lunacek_deterministic(
    fn: Callable, ndim: int, key: PRNGKeyArray | None = None
) -> BBOBFn:
    """Deterministic factory for lunacek_bi_rastrigin."""
    partial_fn, f_opt_val = make_determinstic(fn, ndim)
    kw = _partial_keywords(partial_fn)
    mat = kw["Q"] @ lambda_func(ndim, alpha=100.0) @ kw["R"]
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    mu0 = 2.5
    d = 1.0
    x_opt_shape = (mu0 / 2.0) * bernoulli_vector(ndim, rng)
    s = 1.0 - 1.0 / (2.0 * jnp.sqrt(ndim + 20.0) - 8.2)
    mu1 = -jnp.sqrt((mu0**2 - d) / s)
    return Partial(
        fn,
        **kw,
        _mat=mat,
        _x_opt_shape=x_opt_shape,
        _s=s,
        _mu1=mu1,
    ), f_opt_val


def _make_gallagher_randomized(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    num_peaks: int,
    w_divisor: int,
    alpha_first: float,
    y_minval: float,
    y_maxval: float,
) -> BBOBFn:
    """Randomized factory for Gallagher peak functions."""
    partial_fn, f_opt_val = make_randomized(fn, ndim, key)
    kw = _partial_keywords(partial_fn)
    w, y, c_diags = _precompute_gallagher(
        kw["x_opt"],
        kw["Q"],
        ndim,
        num_peaks,
        w_divisor,
        alpha_first,
        y_minval,
        y_maxval,
    )
    return Partial(
        fn,
        x_opt=kw["x_opt"],
        f_opt=kw["f_opt"],
        R=kw["R"],
        Q=kw["Q"],
        _gal_w=w,
        _gal_y=y,
        _gal_c_diags=c_diags,
    ), f_opt_val


def _make_gallagher_deterministic(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_peaks: int = 101,
    w_divisor: int = 99,
    alpha_first: float = 1000.0,
    y_minval: float = -5.0,
    y_maxval: float = 5.0,
) -> BBOBFn:
    """Deterministic factory for Gallagher peak functions."""
    partial_fn, f_opt_val = make_determinstic(fn, ndim)
    kw = _partial_keywords(partial_fn)
    w, y, c_diags = _precompute_gallagher(
        kw["x_opt"],
        kw["Q"],
        ndim,
        num_peaks,
        w_divisor,
        alpha_first,
        y_minval,
        y_maxval,
    )
    return Partial(
        fn,
        x_opt=kw["x_opt"],
        f_opt=kw["f_opt"],
        R=kw["R"],
        Q=kw["Q"],
        _gal_w=w,
        _gal_y=y,
        _gal_c_diags=c_diags,
    ), f_opt_val


# =============================================================================

registry: dict[str, Callable[[int, PRNGKeyArray], BBOBFn]] = {
    "attractive_sector": Partial(
        _make_with_mat, fn=attractive_sector, alpha=10.0, order="QLR"
    ),
    "bent_cigar": Partial(make_randomized, fn=bent_cigar),
    "discuss": Partial(make_randomized, fn=discuss),
    "ellipsoid": Partial(make_randomized, fn=ellipsoid),
    "ellipsoid_seperable": Partial(make_randomized, fn=ellipsoid_seperable),
    "gallagher_21_peaks": Partial(
        _make_gallagher_randomized,
        fn=gallagher_21_peaks,
        num_peaks=21,
        w_divisor=19,
        alpha_first=1000.0**2,
        y_minval=-4.9,
        y_maxval=4.9,
    ),
    "gallagher_101_peaks": Partial(
        _make_gallagher_randomized,
        fn=gallagher_101_peaks,
        num_peaks=101,
        w_divisor=99,
        alpha_first=1000.0,
        y_minval=-5.0,
        y_maxval=5.0,
    ),
    "griewank_rosenbrock_f8f2": Partial(
        make_randomized, fn=griewank_rosenbrock_f8f2
    ),
    "katsuura": Partial(_make_with_mat, fn=katsuura, alpha=100.0, order="QLR"),
    "linear_slope": Partial(_make_linear_slope, fn=linear_slope),
    "lunacek_bi_rastrigin": Partial(
        _make_lunacek_randomized, fn=lunacek_bi_rastrigin
    ),
    "rastrigin": Partial(
        _make_with_mat, fn=rastrigin, alpha=10.0, order="RLQ"
    ),
    "rastrigin_seperable": Partial(make_randomized, fn=rastrigin_seperable),
    "rosenbrock": Partial(make_randomized, fn=rosenbrock),
    "rosenbrock_rotated": Partial(make_randomized, fn=rosenbrock_rotated),
    "schaffer_f7_condition_10": Partial(
        _make_with_mat, fn=schaffer_f7_condition_10, alpha=10.0, order="LQ"
    ),
    "schaffer_f7_condition_1000": Partial(
        _make_with_mat,
        fn=schaffer_f7_condition_1000,
        alpha=1000.0,
        order="LQ",
    ),
    "schwefel_xsinx": Partial(_make_schwefel, fn=schwefel_xsinx),
    "sharp_ridge": Partial(
        _make_with_mat, fn=sharp_ridge, alpha=10.0, order="QLR"
    ),
    "skew_rastrigin_bueche": Partial(
        make_randomized, fn=skew_rastrigin_bueche
    ),
    "sphere": Partial(make_randomized, fn=sphere),
    "step_ellipsoid": Partial(
        _make_with_mat, fn=step_ellipsoid, alpha=10.0, order="LR"
    ),
    "sum_of_different_powers": Partial(
        make_randomized, fn=sum_of_different_powers
    ),
    "weierstrass": Partial(
        _make_with_mat, fn=weierstrass, alpha=0.01, order="RLQ"
    ),
}

registry_original: dict[str, Callable[[int], BBOBFn]] = {
    "attractive_sector": Partial(
        _make_with_mat_deterministic,
        fn=attractive_sector,
        alpha=10.0,
        order="QLR",
    ),
    "bent_cigar": Partial(make_determinstic, fn=bent_cigar),
    "discuss": Partial(make_determinstic, fn=discuss),
    "ellipsoid": Partial(make_determinstic, fn=ellipsoid),
    "ellipsoid_seperable": Partial(make_determinstic, fn=ellipsoid_seperable),
    "gallagher_21_peaks": Partial(
        _make_gallagher_deterministic,
        fn=gallagher_21_peaks,
        num_peaks=21,
        w_divisor=19,
        alpha_first=1000.0**2,
        y_minval=-4.9,
        y_maxval=4.9,
    ),
    "gallagher_101_peaks": Partial(
        _make_gallagher_deterministic,
        fn=gallagher_101_peaks,
        num_peaks=101,
        w_divisor=99,
        alpha_first=1000.0,
        y_minval=-5.0,
        y_maxval=5.0,
    ),
    "griewank_rosenbrock_f8f2": Partial(
        make_determinstic, fn=griewank_rosenbrock_f8f2
    ),
    "katsuura": Partial(
        _make_with_mat_deterministic,
        fn=katsuura,
        alpha=100.0,
        order="QLR",
    ),
    "linear_slope": Partial(_make_linear_slope_deterministic, fn=linear_slope),
    "lunacek_bi_rastrigin": Partial(
        _make_lunacek_deterministic, fn=lunacek_bi_rastrigin
    ),
    "rastrigin": Partial(
        _make_with_mat_deterministic,
        fn=rastrigin,
        alpha=10.0,
        order="RLQ",
    ),
    "rastrigin_seperable": Partial(make_determinstic, fn=rastrigin_seperable),
    "rosenbrock": Partial(make_determinstic, fn=rosenbrock),
    "rosenbrock_rotated": Partial(make_determinstic, fn=rosenbrock_rotated),
    "schaffer_f7_condition_10": Partial(
        _make_with_mat_deterministic,
        fn=schaffer_f7_condition_10,
        alpha=10.0,
        order="LQ",
    ),
    "schaffer_f7_condition_1000": Partial(
        _make_with_mat_deterministic,
        fn=schaffer_f7_condition_1000,
        alpha=1000.0,
        order="LQ",
    ),
    "schwefel_xsinx": Partial(_make_schwefel_deterministic, fn=schwefel_xsinx),
    "sharp_ridge": Partial(
        _make_with_mat_deterministic,
        fn=sharp_ridge,
        alpha=10.0,
        order="QLR",
    ),
    "skew_rastrigin_bueche": Partial(
        make_determinstic, fn=skew_rastrigin_bueche
    ),
    "sphere": Partial(make_determinstic, fn=sphere),
    "step_ellipsoid": Partial(
        _make_with_mat_deterministic,
        fn=step_ellipsoid,
        alpha=10.0,
        order="LR",
    ),
    "sum_of_different_powers": Partial(
        make_determinstic, fn=sum_of_different_powers
    ),
    "weierstrass": Partial(
        _make_with_mat_deterministic,
        fn=weierstrass,
        alpha=0.01,
        order="RLQ",
    ),
}


def make_randomized_cec2005(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    num_components: int = 1,
    minval: float = -100.0,
    maxval: float = 100.0,
) -> BBOBFn:
    """Factory for CEC 2005 functions with seed-generated params.

    Splits key into ``2*num_components+2`` subkeys so x_opt,
    R, Q, and f_opt consume distinct subkeys (avoids the
    BBOB key-reuse pattern).

    Parameters
    ----------
    fn : Callable
        Base CEC 2005 function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray
        JAX random key.
    num_components : int, optional
        Number of composition components (default 1).
    minval : float, optional
        Lower bound for x_opt (default -100).
    maxval : float, optional
        Upper bound for x_opt (default 100).

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    total_keys = 2 * num_components + 2
    keys = jr.split(key, total_keys)
    # keys[0:num_components]              → R matrices
    # keys[num_components:2*num_components] → Q matrices
    # keys[-2]                            → x_opt seed
    # keys[-1]                            → f_opt seed

    if num_components == 1:
        x_opt = xopt(key=keys[-2], ndim=ndim, minval=minval, maxval=maxval)
        R = rotation_matrix(ndim, keys[0])
        Q = rotation_matrix(ndim, keys[num_components])
    else:
        xopt_keys = jr.split(keys[-2], num_components)
        x_opt = jnp.stack(
            [
                xopt(key=xopt_keys[i], ndim=ndim, minval=minval, maxval=maxval)
                for i in range(num_components)
            ]
        )
        R = jnp.stack(
            [rotation_matrix(ndim, keys[i]) for i in range(num_components)]
        )
        Q = jnp.stack(
            [
                rotation_matrix(ndim, keys[num_components + i])
                for i in range(num_components)
            ]
        )

    f_opt_val = fopt(keys[-1])
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=R, Q=Q), f_opt_val


def make_randomized_cec2005_conditioned(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    num_components: int = 1,
    minval: float = -100.0,
    maxval: float = 100.0,
    condition_numbers: jax.Array | None = None,
) -> BBOBFn:
    """Factory for CEC 2005 functions with conditioned transforms.

    Wraps ``make_randomized_cec2005`` and replaces rotation
    matrices with conditioned linear transforms when
    ``condition_numbers`` is provided.

    Parameters
    ----------
    fn : Callable
        Base CEC 2005 function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray
        JAX random key.
    num_components : int, optional
        Number of composition components (default 1).
    minval : float, optional
        Lower bound for x_opt (default -100).
    maxval : float, optional
        Upper bound for x_opt (default 100).
    condition_numbers : jax.Array or None, optional
        Target condition numbers per component.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    partial_fn, f_opt_val = make_randomized_cec2005(
        fn, ndim, key, num_components, minval=minval, maxval=maxval
    )
    partial_keywords = _partial_keywords(partial_fn)
    x_opt = partial_keywords["x_opt"]
    q = partial_keywords["Q"]
    r_key = jr.fold_in(key, 11)
    if condition_numbers is None:
        r = partial_keywords["R"]
    elif num_components == 1:
        r = _conditioned_linear_transform(
            ndim, r_key, float(jnp.asarray(condition_numbers))
        )
    else:
        conds = jnp.asarray(condition_numbers, dtype=jnp.float32)
        r_keys = jr.split(r_key, num_components)
        r = _conditioned_transform_stack(ndim, r_keys, conds)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=r, Q=q), f_opt_val


def make_deterministic_cec2005(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
) -> BBOBFn:
    """Deterministic CEC 2005 factory with zero shift.

    Uses identity rotations. ``key`` is accepted and ignored
    so both registries have identical signatures.

    Parameters
    ----------
    fn : Callable
        Base CEC 2005 function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        Ignored; accepted for signature compatibility.
    num_components : int, optional
        Number of composition components (default 1).

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    f_opt_val = jnp.array(0.0)
    if num_components == 1:
        x_opt = jnp.zeros(ndim)
        eye = jnp.eye(ndim)
        return Partial(
            fn, x_opt=x_opt, f_opt=f_opt_val, R=eye, Q=eye
        ), f_opt_val
    else:
        x_opt = jnp.zeros((num_components, ndim))
        eyes = jnp.stack([jnp.eye(ndim)] * num_components)
        return (
            Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=eyes, Q=eyes),
            f_opt_val,
        )


def _make_randomized_cec2005_f5(
    fn: Callable, ndim: int, key: PRNGKeyArray, num_components: int = 1
) -> BBOBFn:
    """F5: integer A in [-500, 500], optimum clamped to the bounds."""
    total_keys = 2 * num_components + 2
    keys = jr.split(key, total_keys)
    x_opt = xopt(key=keys[-2], ndim=ndim, minval=-100.0, maxval=100.0)
    k = math.ceil(ndim / 4)
    x_opt = x_opt.at[:k].set(-100.0)
    x_opt = x_opt.at[-k:].set(100.0)
    a = _sample_nonsingular_integer_matrix(keys[0], ndim, -500, 500)
    f_opt_val = fopt(keys[-1])
    return Partial(
        fn,
        x_opt=x_opt,
        f_opt=f_opt_val,
        R=a,
        Q=jnp.eye(ndim),
    ), f_opt_val


def _make_randomized_cec2005_f8(
    fn: Callable, ndim: int, key: PRNGKeyArray, num_components: int = 1
) -> BBOBFn:
    """Factory for F8: boundary optimum and condition-100 transform."""
    partial_fn, f_opt_val = make_randomized_cec2005_conditioned(
        fn,
        ndim,
        key,
        num_components,
        minval=-32.0,
        maxval=32.0,
        condition_numbers=jnp.array(100.0),
    )
    partial_keywords = _partial_keywords(partial_fn)
    x_opt = partial_keywords["x_opt"]
    odd_1based_indices = jnp.arange(0, ndim, 2)
    x_opt = x_opt.at[odd_1based_indices].set(-32.0)
    return Partial(
        fn,
        x_opt=x_opt,
        f_opt=f_opt_val,
        R=partial_keywords["R"],
        Q=partial_keywords["Q"],
    ), f_opt_val


def _make_randomized_cec2005_f12(
    fn: Callable, ndim: int, key: PRNGKeyArray, num_components: int = 1
) -> BBOBFn:
    """Factory for F12: integer A/B matrices, alpha in [-pi, pi]."""
    total_keys = 2 * num_components + 2
    keys = jr.split(key, total_keys)
    x_opt = xopt(key=keys[-2], ndim=ndim, minval=-math.pi, maxval=math.pi)
    # A and B matrices: integer-valued in [-100, 100]
    R = jr.randint(keys[0], (ndim, ndim), -100, 101).astype(jnp.float32)
    Q = jr.randint(keys[num_components], (ndim, ndim), -100, 101).astype(
        jnp.float32
    )
    f_opt_val = fopt(keys[-1])
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=R, Q=Q), f_opt_val


def _make_randomized_cec2005_f7(
    fn: Callable, ndim: int, key: PRNGKeyArray, num_components: int = 1
) -> BBOBFn:
    """Factory for F7: optimum outside init range, scaled cond-3 transform."""
    keys = jr.split(key, 4)
    x_opt = xopt(key=keys[2], ndim=ndim, minval=-100.0, maxval=0.0)
    base = _conditioned_linear_transform(ndim, keys[0], 3.0)
    scale = 1.0 + 0.3 * jnp.abs(jr.normal(keys[1], shape=()))
    r = base * scale
    q = jnp.eye(ndim)
    f_opt_val = fopt(keys[3])
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=r, Q=q), f_opt_val


def _make_randomized_cec2005_conditioned_single(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    *,
    minval: float,
    maxval: float,
    condition_number: float,
) -> BBOBFn:
    """Single-component conditioned CEC 2005 factory."""
    return make_randomized_cec2005_conditioned(
        fn,
        ndim,
        key,
        num_components=1,
        minval=minval,
        maxval=maxval,
        condition_numbers=jnp.array(condition_number, dtype=jnp.float32),
    )


def _make_randomized_cec2005_f18_family(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    num_components: int = 10,
) -> BBOBFn:
    """Factory for F18/F19/F20 with paper condition numbers and o10=0."""
    conds = jnp.array([2, 3, 2, 3, 2, 3, 20, 30, 200, 300], dtype=jnp.float32)
    partial_fn, f_opt_val = make_randomized_cec2005_conditioned(
        fn,
        ndim,
        key,
        num_components,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=conds,
    )
    partial_keywords = _partial_keywords(partial_fn)
    x_opt = partial_keywords["x_opt"].at[9].set(jnp.zeros(ndim))
    return Partial(
        fn,
        x_opt=x_opt,
        f_opt=f_opt_val,
        R=partial_keywords["R"],
        Q=partial_keywords["Q"],
    ), f_opt_val


def _make_randomized_cec2005_f20(
    fn: Callable, ndim: int, key: PRNGKeyArray, num_components: int = 10
) -> BBOBFn:
    """Factory for F20: even 1-based coordinates of o1 are fixed to 5."""
    partial_fn, f_opt_val = _make_randomized_cec2005_f18_family(
        fn, ndim, key, num_components
    )
    partial_keywords = _partial_keywords(partial_fn)
    x_opt = partial_keywords["x_opt"]
    even_indices = jnp.arange(1, ndim, 2)
    x_opt = x_opt.at[0, even_indices].set(5.0)
    return Partial(
        fn,
        x_opt=x_opt,
        f_opt=f_opt_val,
        R=partial_keywords["R"],
        Q=partial_keywords["Q"],
    ), f_opt_val


_NC = 10  # num_components for all composition functions (F15-F25)

# Lambda arrays shared by composition function groups
_COMP1_LAMBDA = jnp.array(
    [1, 1, 10, 10, 5 / 60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100]
)
_COMP2_LAMBDA_F18 = jnp.array(
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
_COMP2_LAMBDA_F19 = jnp.array(
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
_COMP3_LAMBDA = jnp.array(
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


def _add_f_max(
    base_factory: Callable,
    comp_fns_builder: Callable,
    comp_lambda: jax.Array,
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray,
    **factory_kwargs: Any,
) -> BBOBFn:
    """Wrap any CEC2005 factory to add precomputed _f_max."""
    partial_fn, f_opt_val = base_factory(fn, ndim, key, **factory_kwargs)
    kw = _partial_keywords(partial_fn)
    M = kw["R"]
    # F15 uses identity stacks, but M is already set correctly by the factory
    fns = comp_fns_builder()
    f_max = compute_composition_f_max(fns, comp_lambda, M, ndim)
    return Partial(fn, **kw, _f_max=f_max), f_opt_val


def _add_f_max_deterministic(
    comp_fns_builder: Callable,
    comp_lambda: jax.Array,
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = _NC,
) -> BBOBFn:
    """Deterministic factory with precomputed _f_max."""
    partial_fn, f_opt_val = make_deterministic_cec2005(
        fn, ndim, key, num_components
    )
    kw = _partial_keywords(partial_fn)
    M = kw["R"]
    fns = comp_fns_builder()
    f_max = compute_composition_f_max(fns, comp_lambda, M, ndim)
    return Partial(fn, **kw, _f_max=f_max), f_opt_val


cec2005_registry: dict[str, Callable] = {
    "f1": Partial(
        make_randomized_cec2005,
        fn=f1,
        num_components=1,
        minval=-100.0,
        maxval=100.0,
    ),
    "f2": Partial(
        make_randomized_cec2005,
        fn=f2,
        num_components=1,
        minval=-100.0,
        maxval=100.0,
    ),
    "f3": Partial(
        make_randomized_cec2005,
        fn=f3,
        num_components=1,
        minval=-100.0,
        maxval=100.0,
    ),
    "f4": Partial(
        make_randomized_cec2005,
        fn=f4,
        num_components=1,
        minval=-100.0,
        maxval=100.0,
    ),
    "f5": Partial(_make_randomized_cec2005_f5, fn=f5, num_components=1),
    "f6": Partial(
        make_randomized_cec2005,
        fn=f6,
        num_components=1,
        minval=-100.0,
        maxval=100.0,
    ),
    "f7": Partial(_make_randomized_cec2005_f7, fn=f7, num_components=1),
    "f8": Partial(_make_randomized_cec2005_f8, fn=f8, num_components=1),
    "f9": Partial(
        make_randomized_cec2005,
        fn=f9,
        num_components=1,
        minval=-5.0,
        maxval=5.0,
    ),
    "f10": Partial(
        _make_randomized_cec2005_conditioned_single,
        fn=f10,
        minval=-5.0,
        maxval=5.0,
        condition_number=2.0,
    ),
    "f11": Partial(
        _make_randomized_cec2005_conditioned_single,
        fn=f11,
        minval=-0.5,
        maxval=0.5,
        condition_number=5.0,
    ),
    "f12": Partial(_make_randomized_cec2005_f12, fn=f12, num_components=1),
    "f13": Partial(
        make_randomized_cec2005,
        fn=f13,
        num_components=1,
        minval=-3.0,
        maxval=1.0,
    ),
    "f14": Partial(
        _make_randomized_cec2005_conditioned_single,
        fn=f14,
        minval=-100.0,
        maxval=100.0,
        condition_number=3.0,
    ),
    "f15": Partial(
        _add_f_max,
        base_factory=make_randomized_cec2005,
        comp_fns_builder=_composition1_fns,
        comp_lambda=_COMP1_LAMBDA,
        fn=f15,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
    ),
    "f16": Partial(
        _add_f_max,
        base_factory=make_randomized_cec2005_conditioned,
        comp_fns_builder=_composition1_fns,
        comp_lambda=_COMP1_LAMBDA,
        fn=f16,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=jnp.full((_NC,), 2.0, dtype=jnp.float32),
    ),
    "f17": Partial(
        _add_f_max,
        base_factory=make_randomized_cec2005,
        comp_fns_builder=_composition1_fns,
        comp_lambda=_COMP1_LAMBDA,
        fn=f17,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
    ),
    "f18": Partial(
        _add_f_max,
        base_factory=_make_randomized_cec2005_f18_family,
        comp_fns_builder=_composition2_fns,
        comp_lambda=_COMP2_LAMBDA_F18,
        fn=f18,
        num_components=_NC,
    ),
    "f19": Partial(
        _add_f_max,
        base_factory=_make_randomized_cec2005_f18_family,
        comp_fns_builder=_composition2_fns,
        comp_lambda=_COMP2_LAMBDA_F19,
        fn=f19,
        num_components=_NC,
    ),
    "f20": Partial(
        _add_f_max,
        base_factory=_make_randomized_cec2005_f20,
        comp_fns_builder=_composition2_fns,
        comp_lambda=_COMP2_LAMBDA_F18,
        fn=f20,
        num_components=_NC,
    ),
    "f21": Partial(
        _add_f_max,
        base_factory=make_randomized_cec2005_conditioned,
        comp_fns_builder=_composition3_fns,
        comp_lambda=_COMP3_LAMBDA,
        fn=f21,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=jnp.ones((_NC,), dtype=jnp.float32),
    ),
    "f22": Partial(
        _add_f_max,
        base_factory=make_randomized_cec2005_conditioned,
        comp_fns_builder=_composition3_fns,
        comp_lambda=_COMP3_LAMBDA,
        fn=f22,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=jnp.array(
            [10, 20, 50, 100, 200, 1000, 2000, 3000, 4000, 5000],
            dtype=jnp.float32,
        ),
    ),
    "f23": Partial(
        _add_f_max,
        base_factory=make_randomized_cec2005_conditioned,
        comp_fns_builder=_composition3_fns,
        comp_lambda=_COMP3_LAMBDA,
        fn=f23,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=jnp.ones((_NC,), dtype=jnp.float32),
    ),
    "f24": Partial(
        make_randomized_cec2005_conditioned,
        fn=f24,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=jnp.array(
            [100, 50, 30, 10, 5, 5, 4, 3, 2, 2], dtype=jnp.float32
        ),
    ),
    "f25": Partial(
        make_randomized_cec2005_conditioned,
        fn=f25,
        num_components=_NC,
        minval=-5.0,
        maxval=5.0,
        condition_numbers=jnp.array(
            [100, 50, 30, 10, 5, 5, 4, 3, 2, 2], dtype=jnp.float32
        ),
    ),
}

cec2005_registry_original: dict[str, Callable] = {
    "f1": Partial(make_deterministic_cec2005, fn=f1, num_components=1),
    "f2": Partial(make_deterministic_cec2005, fn=f2, num_components=1),
    "f3": Partial(make_deterministic_cec2005, fn=f3, num_components=1),
    "f4": Partial(make_deterministic_cec2005, fn=f4, num_components=1),
    "f5": Partial(make_deterministic_cec2005, fn=f5, num_components=1),
    "f6": Partial(make_deterministic_cec2005, fn=f6, num_components=1),
    "f7": Partial(make_deterministic_cec2005, fn=f7, num_components=1),
    "f8": Partial(make_deterministic_cec2005, fn=f8, num_components=1),
    "f9": Partial(make_deterministic_cec2005, fn=f9, num_components=1),
    "f10": Partial(make_deterministic_cec2005, fn=f10, num_components=1),
    "f11": Partial(make_deterministic_cec2005, fn=f11, num_components=1),
    "f12": Partial(make_deterministic_cec2005, fn=f12, num_components=1),
    "f13": Partial(make_deterministic_cec2005, fn=f13, num_components=1),
    "f14": Partial(make_deterministic_cec2005, fn=f14, num_components=1),
    "f15": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition1_fns,
        comp_lambda=_COMP1_LAMBDA,
        fn=f15,
        num_components=_NC,
    ),
    "f16": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition1_fns,
        comp_lambda=_COMP1_LAMBDA,
        fn=f16,
        num_components=_NC,
    ),
    "f17": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition1_fns,
        comp_lambda=_COMP1_LAMBDA,
        fn=f17,
        num_components=_NC,
    ),
    "f18": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition2_fns,
        comp_lambda=_COMP2_LAMBDA_F18,
        fn=f18,
        num_components=_NC,
    ),
    "f19": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition2_fns,
        comp_lambda=_COMP2_LAMBDA_F19,
        fn=f19,
        num_components=_NC,
    ),
    "f20": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition2_fns,
        comp_lambda=_COMP2_LAMBDA_F18,
        fn=f20,
        num_components=_NC,
    ),
    "f21": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition3_fns,
        comp_lambda=_COMP3_LAMBDA,
        fn=f21,
        num_components=_NC,
    ),
    "f22": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition3_fns,
        comp_lambda=_COMP3_LAMBDA,
        fn=f22,
        num_components=_NC,
    ),
    "f23": Partial(
        _add_f_max_deterministic,
        comp_fns_builder=_composition3_fns,
        comp_lambda=_COMP3_LAMBDA,
        fn=f23,
        num_components=_NC,
    ),
    "f24": Partial(make_deterministic_cec2005, fn=f24, num_components=_NC),
    "f25": Partial(make_deterministic_cec2005, fn=f25, num_components=_NC),
}
