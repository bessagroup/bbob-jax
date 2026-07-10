"""Factory functions that construct benchmark problem instances.

Each maker binds a base function's parameters (``x_opt``,
``f_opt``, ``R``, ``Q`` and any precomputed keyword arguments)
into a :class:`jax.tree_util.Partial` and returns
``(fn, f_opt)``. Every maker takes a ``deterministic`` flag:
``False`` samples the parameters from ``key`` (random shift,
random rotations, random ``f_opt``); ``True`` short-circuits
the sampling and uses the deterministic parameters (zero
shift, identity rotations, zero ``f_opt``). The randomized and
deterministic registries in ``registry.py`` are the two
adapters of this single factory family.
"""

#                                                                       Modules
# =============================================================================

# Standard
import math
from collections.abc import Callable, Sequence
from typing import Any, cast

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray

# Local
from bbob_jax._src.bbob import _precompute_gallagher
from bbob_jax._src.composition import compute_composition_f_max
from bbob_jax._src.sampling import (
    bernoulli_vector,
    fopt,
    rotation_matrix,
    xopt,
)
from bbob_jax._src.transforms import lambda_func

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
    n = jnp.diag(jnp.asarray(condition_number, dtype=float) ** exponents)
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
    ).astype(float)
    for attempt_key in attempt_keys:
        mat = jr.randint(attempt_key, (ndim, ndim), minval, maxval + 1).astype(
            float
        )
        if not jnp.isclose(jnp.linalg.det(mat), 0.0):
            return mat
    return fallback


#                                                                BBOB factories
# =============================================================================


def make_bbob(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    deterministic: bool = False,
) -> BBOBFn:
    """BBOB factory binding shift, rotations and optimal value.

    Parameters
    ----------
    fn : Callable
        Base BBOB function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        JAX random key for parameter generation. Required when
        ``deterministic`` is False; ignored otherwise.
    deterministic : bool, optional
        When True, use zero shift, identity rotations and zero
        ``f_opt`` instead of sampling from ``key``.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    if deterministic:
        x_opt = jnp.zeros(ndim)
        eye = jnp.eye(ndim)
        f_opt = jnp.array(0.0)
        return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=eye, Q=eye), f_opt
    if key is None:
        raise ValueError("key is required when deterministic=False")
    key1, key2 = jr.split(key)
    x_opt = xopt(key=key1, ndim=ndim, minval=-4.0, maxval=4.0)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    f_opt = fopt(key)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=R, Q=Q), f_opt


def _make_with_mat(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    alpha: float = 10.0,
    order: str = "QLR",
    deterministic: bool = False,
) -> BBOBFn:
    """Factory that precomputes a matrix chain.

    Combines ``lambda_func`` conditioning with the rotation
    matrices in the specified ``order``.
    """
    partial_fn, f_opt_val = make_bbob(fn, ndim, key, deterministic)
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


def _make_linear_slope(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for linear_slope with precomputed random state."""
    partial_fn, f_opt_val = make_bbob(fn, ndim, key, deterministic)
    kw = _partial_keywords(partial_fn)
    rng = jr.key(0)
    rng = jr.fold_in(rng, kw["Q"][0, 0])
    ls_x_opt = 5 * bernoulli_vector(ndim, rng)
    i = jnp.arange(1, ndim + 1, dtype=float)
    ls_s = jnp.sign(ls_x_opt) * jnp.power(10.0, (i - 1) / (ndim - 1))
    return Partial(fn, **kw, _ls_x_opt=ls_x_opt, _ls_s=ls_s), f_opt_val


def _make_schwefel(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for schwefel_xsinx with precomputed random state + lambda."""
    partial_fn, f_opt_val = make_bbob(fn, ndim, key, deterministic)
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


def _make_lunacek(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for lunacek_bi_rastrigin with precomputed state."""
    partial_fn, f_opt_val = make_bbob(fn, ndim, key, deterministic)
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


def _make_gallagher(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_peaks: int = 101,
    w_divisor: int = 99,
    alpha_first: float = 1000.0,
    y_minval: float = -5.0,
    y_maxval: float = 5.0,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for Gallagher peak functions."""
    partial_fn, f_opt_val = make_bbob(fn, ndim, key, deterministic)
    kw = _partial_keywords(partial_fn)
    w, y_rot, c_diags = _precompute_gallagher(
        kw["x_opt"],
        kw["R"],
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
        _gal_y_rot=y_rot,
        _gal_c_diags=c_diags,
    ), f_opt_val


#                                                            CEC 2005 factories
# =============================================================================


def make_cec2005(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
    minval: float = -100.0,
    maxval: float = 100.0,
    deterministic: bool = False,
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
    key : PRNGKeyArray or None, optional
        JAX random key. Required when ``deterministic`` is
        False; ignored otherwise.
    num_components : int, optional
        Number of composition components (default 1).
    minval : float, optional
        Lower bound for x_opt (default -100).
    maxval : float, optional
        Upper bound for x_opt (default 100).
    deterministic : bool, optional
        When True, use zero shift, identity rotations and zero
        ``f_opt`` instead of sampling from ``key``.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    if deterministic:
        f_opt_val = jnp.array(0.0)
        if num_components == 1:
            x_opt = jnp.zeros(ndim)
            eye = jnp.eye(ndim)
            return Partial(
                fn, x_opt=x_opt, f_opt=f_opt_val, R=eye, Q=eye
            ), f_opt_val
        x_opt = jnp.zeros((num_components, ndim))
        eyes = jnp.stack([jnp.eye(ndim)] * num_components)
        return (
            Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=eyes, Q=eyes),
            f_opt_val,
        )
    if key is None:
        raise ValueError("key is required when deterministic=False")

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


def make_cec2017(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    min_ndim: int = 1,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for CEC 2017 functions with seed-generated params.

    Delegates to :func:`make_cec2005` (same sampling scheme:
    distinct subkeys for R, Q, x_opt and f_opt) with the shift
    sampled in ``[-80, 80]^D`` as in the official suite, where
    shifts stay inside 80% of the ``[-100, 100]`` search range.

    Parameters
    ----------
    fn : Callable
        Base CEC 2017 function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        JAX random key. Required when ``deterministic`` is
        False; ignored otherwise.
    min_ndim : int, optional
        Smallest ``ndim`` the function is defined for
        (default 1).
    deterministic : bool, optional
        When True, use zero shift, identity rotations and zero
        ``f_opt`` instead of sampling from ``key``.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).

    Raises
    ------
    ValueError
        If ``ndim < min_ndim``.
    """
    if ndim < min_ndim:
        raise ValueError(
            f"{getattr(fn, '__name__', fn)} requires ndim >= {min_ndim}, "
            f"got {ndim}"
        )
    return make_cec2005(
        fn,
        ndim,
        key,
        num_components=1,
        minval=-80.0,
        maxval=80.0,
        deterministic=deterministic,
    )


def make_cec2017_hybrid(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    min_ndim: int = 1,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for CEC 2017 hybrid functions (F11-F20).

    Beyond the shift (sampled in ``[-80, 80]^D``), rotations and
    ``f_opt``, hybrids carry a dimension permutation bound as
    ``_shuffle`` (the official suite ships shuffle data files;
    here it is sampled from the key). ``deterministic=True``
    uses the identity permutation alongside zero shift, identity
    rotations and zero ``f_opt``.

    Parameters
    ----------
    fn : Callable
        Base CEC 2017 hybrid function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        JAX random key. Required when ``deterministic`` is
        False; ignored otherwise.
    min_ndim : int, optional
        Smallest ``ndim`` the hybrid is defined for (at least
        one dimension per subcomponent kernel; more when the
        chunk split demands it — see
        ``cec2017_hybrid_partition``).
    deterministic : bool, optional
        When True, use zero shift, identity rotations, identity
        shuffle and zero ``f_opt`` instead of sampling.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).

    Raises
    ------
    ValueError
        If ``ndim < min_ndim``.
    """
    if ndim < min_ndim:
        raise ValueError(
            f"{getattr(fn, '__name__', fn)} requires ndim >= {min_ndim}, "
            f"got {ndim}"
        )
    if deterministic:
        f_opt_val = jnp.array(0.0)
        eye = jnp.eye(ndim)
        return (
            Partial(
                fn,
                x_opt=jnp.zeros(ndim),
                f_opt=f_opt_val,
                R=eye,
                Q=eye,
                _shuffle=jnp.arange(ndim),
            ),
            f_opt_val,
        )
    if key is None:
        raise ValueError("key is required when deterministic=False")

    key_r, key_q, key_s, key_x, key_f = jr.split(key, 5)
    x_opt = xopt(key=key_x, ndim=ndim, minval=-80.0, maxval=80.0)
    R = rotation_matrix(ndim, key_r)
    Q = rotation_matrix(ndim, key_q)
    shuffle = jr.permutation(key_s, ndim)
    f_opt_val = fopt(key_f)
    return (
        Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=R, Q=Q, _shuffle=shuffle),
        f_opt_val,
    )


def make_cec2005_conditioned(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
    minval: float = -100.0,
    maxval: float = 100.0,
    condition_numbers: jax.Array | Sequence[float] | None = None,
    deterministic: bool = False,
) -> BBOBFn:
    """Factory for CEC 2005 functions with conditioned transforms.

    Wraps ``make_cec2005`` and replaces rotation matrices with
    conditioned linear transforms when ``condition_numbers``
    is provided. In deterministic mode the conditioning is
    skipped and the identity transforms are kept.

    Parameters
    ----------
    fn : Callable
        Base CEC 2005 function.
    ndim : int
        Number of input dimensions.
    key : PRNGKeyArray or None, optional
        JAX random key. Required when ``deterministic`` is
        False; ignored otherwise.
    num_components : int, optional
        Number of composition components (default 1).
    minval : float, optional
        Lower bound for x_opt (default -100).
    maxval : float, optional
        Upper bound for x_opt (default 100).
    condition_numbers : jax.Array, sequence of float, or None, optional
        Target condition numbers per component.
    deterministic : bool, optional
        When True, use zero shift, identity rotations and zero
        ``f_opt`` instead of sampling from ``key``.

    Returns
    -------
    BBOBFn
        Tuple of (partial function, optimal value).
    """
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    partial_fn, f_opt_val = make_cec2005(
        fn, ndim, key, num_components, minval=minval, maxval=maxval
    )
    partial_keywords = _partial_keywords(partial_fn)
    x_opt = partial_keywords["x_opt"]
    q = partial_keywords["Q"]
    assert key is not None
    r_key = jr.fold_in(key, 11)
    if condition_numbers is None:
        r = partial_keywords["R"]
    elif num_components == 1:
        r = _conditioned_linear_transform(
            ndim, r_key, float(jnp.asarray(condition_numbers))
        )
    else:
        conds = jnp.asarray(condition_numbers, dtype=float)
        r_keys = jr.split(r_key, num_components)
        r = _conditioned_transform_stack(ndim, r_keys, conds)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=r, Q=q), f_opt_val


def _make_cec2005_conditioned_single(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    *,
    minval: float,
    maxval: float,
    condition_number: float,
    deterministic: bool = False,
) -> BBOBFn:
    """Single-component conditioned CEC 2005 factory."""
    return make_cec2005_conditioned(
        fn,
        ndim,
        key,
        num_components=1,
        minval=minval,
        maxval=maxval,
        condition_numbers=jnp.array(condition_number, dtype=float),
        deterministic=deterministic,
    )


def _make_cec2005_f5(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
    deterministic: bool = False,
) -> BBOBFn:
    """F5: integer A in [-500, 500], optimum clamped to the bounds."""
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    assert key is not None
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


def _make_cec2005_f7(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
    deterministic: bool = False,
) -> BBOBFn:
    """F7: optimum outside init range, scaled cond-3 transform."""
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    assert key is not None
    keys = jr.split(key, 4)
    x_opt = xopt(key=keys[2], ndim=ndim, minval=-100.0, maxval=0.0)
    base = _conditioned_linear_transform(ndim, keys[0], 3.0)
    scale = 1.0 + 0.3 * jnp.abs(jr.normal(keys[1], shape=()))
    r = base * scale
    q = jnp.eye(ndim)
    f_opt_val = fopt(keys[3])
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=r, Q=q), f_opt_val


def _make_cec2005_f8(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
    deterministic: bool = False,
) -> BBOBFn:
    """F8: boundary optimum and condition-100 transform."""
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    partial_fn, f_opt_val = make_cec2005_conditioned(
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


def _make_cec2005_f12(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
    deterministic: bool = False,
) -> BBOBFn:
    """F12: integer A/B matrices, alpha in [-pi, pi]."""
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    assert key is not None
    total_keys = 2 * num_components + 2
    keys = jr.split(key, total_keys)
    x_opt = xopt(key=keys[-2], ndim=ndim, minval=-math.pi, maxval=math.pi)
    # A and B matrices: integer-valued in [-100, 100]
    R = jr.randint(keys[0], (ndim, ndim), -100, 101).astype(float)
    Q = jr.randint(keys[num_components], (ndim, ndim), -100, 101).astype(float)
    f_opt_val = fopt(keys[-1])
    return Partial(fn, x_opt=x_opt, f_opt=f_opt_val, R=R, Q=Q), f_opt_val


def _make_cec2005_f18_family(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 10,
    deterministic: bool = False,
) -> BBOBFn:
    """F18/F19/F20 with paper condition numbers and o10=0."""
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    conds = jnp.array([2, 3, 2, 3, 2, 3, 20, 30, 200, 300], dtype=float)
    partial_fn, f_opt_val = make_cec2005_conditioned(
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


def _make_cec2005_f20(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 10,
    deterministic: bool = False,
) -> BBOBFn:
    """F20: even 1-based coordinates of o1 are fixed to 5."""
    if deterministic:
        return make_cec2005(fn, ndim, key, num_components, deterministic=True)
    partial_fn, f_opt_val = _make_cec2005_f18_family(
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


def _add_f_max(
    base_factory: Callable,
    comp_fns_builder: Callable,
    comp_lambda: jax.Array | Sequence[float],
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    deterministic: bool = False,
    **factory_kwargs: Any,
) -> BBOBFn:
    """Wrap any CEC 2005 factory to add precomputed _f_max.

    ``comp_lambda`` may be a ``jax.Array`` or any sequence of floats;
    it is converted to ``jax.Array`` at call time so registry entries
    can store dtype-free Python tuples (see the ``_COMP*_LAMBDA``
    constants in ``spec.py``).
    """
    partial_fn, f_opt_val = base_factory(
        fn, ndim, key, deterministic=deterministic, **factory_kwargs
    )
    kw = _partial_keywords(partial_fn)
    M = kw["R"]
    # F15 uses identity stacks, but M is already set correctly by the factory
    fns = comp_fns_builder()
    lambda_arr = jnp.asarray(comp_lambda, dtype=float)
    f_max = compute_composition_f_max(fns, lambda_arr, M, ndim)
    return Partial(fn, **kw, _f_max=f_max), f_opt_val
