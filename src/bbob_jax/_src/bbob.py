#                                                                       Modules
# =============================================================================

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

# Local
from bbob_jax._src.registry import register_function
from bbob_jax._src.utils import (bernoulli_vector, lambda_func, penalty,
                                 rotation_matrix, tasy_func, tosz_func)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


@register_function('sphere')
def sphere(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    z = x - x_opt
    return jnp.sum(z ** 2)


@register_function('ellipsoid_seperable')
def ellipsoid_seperable(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    idx = jnp.arange(ndim, dtype=x.dtype)
    z = tosz_func(x - x_opt)
    weights = 10.0 ** (6.0 * idx / (ndim - 1))
    return jnp.sum(weights * z**2)


@register_function('rastrigin_seperable')
def rastrigin_seperable(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]

    alpha = lambda_func(ndim, alpha=10.0)
    temp = tosz_func(x - x_opt)
    z = jnp.matmul(alpha, tasy_func(temp, beta=0.2))

    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0 * ndim)


@register_function('skew_rastrigin_bueche')
def skew_rastrigin_bueche(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array):
    ndim = x.shape[-1]
    d = jnp.arange(ndim, dtype=x.dtype)
    s = 10 ** (0.5 * (d - 1) / (ndim - 1))
    odd_indices = jnp.arange(0, ndim, 2)
    z = s[None, :] * tosz_func(x - x_opt)

    # Modify odd indices
    z_odd = jnp.where(z[:, odd_indices] > 0,
                      z[:, odd_indices] * 10, z[:, odd_indices])
    z = z.at[:, odd_indices].set(z_odd)

    # Compute terms
    first_part = 10 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z), axis=-1))
    second_part = jnp.sum(z * z, axis=-1)

    y = first_part + second_part
    return y


@register_function('linear_slope')
def linear_slope(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    x_opt = 5 * bernoulli_vector(ndim, key)
    s = jnp.sign(x_opt) * jnp.power(10.0, jnp.arange(1, ndim + 1) / (ndim - 1))

    cond = x * x_opt < 25.0
    z = jnp.where(cond, x, x_opt)

    result = jnp.sum(5.0 * jnp.abs(s) - s * z)
    return result


@register_function('attractive_sector')
def attractive_sector(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    key1, key2 = jr.split(key)
    ndim = x.shape[-1]
    cond = (x_opt * x) > 0.0
    s = jnp.where(cond, 100.0, 1.0)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=10.0)
    z = Q @ lamb @ R @ (x - x_opt)

    term = jnp.sum((s * z) ** 2)

    result = jnp.power(tosz_func(term), 0.9)

    return result


@register_function('step_ellipsoid')
def step_ellipsoid(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    key1, key2 = jr.split(key)
    ndim = x.shape[-1]
    lamb = lambda_func(ndim, alpha=10.0)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    mult = jnp.power(10.0, 2 * jnp.arange(0, ndim,
                     dtype=jnp.float32) / (ndim - 1))

    # Compute ẑ
    z_hat = lamb @ R @ (x - x_opt)

    # Compute z′ using functional indexing
    z_dash = 0.5 + jnp.where(jnp.abs(z_hat) > 0.5, z_hat, 10 * z_hat)
    z_dash = jnp.floor(z_dash)
    z_dash = jnp.where(jnp.abs(z_hat) > 0.5, z_dash, z_dash / 10.0)

    # Compute z
    z = Q @ z_dash

    # Compute final f
    result = 0.1 * jnp.maximum(
        jnp.abs(z_hat[0]) / 10**4,
        jnp.sum(mult * z * z)
    )
    return result


@register_function('rosenbrock')
def rosenbrock(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
    # Shift and scale
    z = zmax * (x - x_opt) + 1  # shape (..., dim)

    # Create unshifted and shifted arrays along last axis
    unshift = z[..., :-1]  # all except last
    shifted = z[..., 1:]   # all except first

    # Compute the sum
    result = jnp.sum(
        100.0 * jnp.power(unshift**2 - shifted, 2) +
        jnp.power(unshift - 1.0, 2),
        axis=-1
    )

    return result


@register_function('rosenbrock_rotated')
def rosenbrock_rotated(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key)

    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)

    z = zmax * (x @ R) + 0.5

    # Create unshifted and shifted arrays along last axis
    unshift = z[..., :-1]  # all except last
    shifted = z[..., 1:]   # all except first

    # Compute the sum
    result = jnp.sum(
        100.0 * jnp.power(unshift**2 - shifted, 2) +
        jnp.power(unshift - 1.0, 2),
        axis=-1
    )

    return result


@register_function('ellipsoid')
def ellipsoid(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key)
    idx = jnp.arange(ndim, dtype=x.dtype)
    z = tosz_func(x @ R)
    weights = 10.0 ** (6.0 * idx / (ndim - 1))
    return jnp.sum(weights * z**2)


@register_function('discuss')
def discuss(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key)
    z = tosz_func(R @ (x - x_opt))
    first = 1e6 * jnp.power(z[..., 0], 2)
    second = jnp.sum(jnp.power(z[..., 1:], 2), axis=-1)
    return first + second


@register_function('bent_cigar')
def bent_cigar(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key)
    z = tasy_func(R @ (x - x_opt), beta=0.5) @ R
    first = jnp.power(z[..., 0], 2)
    second = 1e6 * jnp.sum(jnp.power(z[..., 1:], 2), axis=-1)
    return first + second


@register_function('sharp_ridge')
def sharp_ridge(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    key1, key2 = jr.split(key)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=10.0)
    z = Q @ lamb @ R @ (x - x_opt)
    first = jnp.power(z[..., 0], 2)
    second = 100.0 * jnp.sqrt(jnp.sum(jnp.power(z[..., 1:], 2), axis=-1))
    return first + second


@register_function('sum_of_different_powers')
def sum_of_different_powers(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key)
    z = R @ (x - x_opt)
    idx = jnp.arange(1, ndim + 1, dtype=x.dtype)
    return jnp.sum(jnp.abs(z) ** (2 + 4 * (idx - 1) / (ndim - 1)))


@register_function('rastrigin')
def rastrigin(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    key1, key2 = jr.split(key)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=10.0)
    z = R @ lamb @ Q @ tasy_func(tosz_func(R @
                                 (x - x_opt)), beta=0.2)

    return jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z) + 10.0 * ndim)


@register_function('weierstrass')
def weierstrass(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    key1, key2 = jr.split(key)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=0.01)
    z = R @ lamb @ Q @ tosz_func(R @ (x - x_opt))

    k = jnp.arange(0, 11)
    ak = 0.5 ** k
    bk = 3.0 ** k

    first_term = jnp.sum(ak[:, None] * jnp.cos(2 * jnp.pi *
                                               bk[:, None] * (z + 0.5)), axis=0)
    correction = jnp.sum(ak * jnp.cos(jnp.pi * bk))

    result = 10.0 * jnp.power(1.0 / ndim * jnp.sum(first_term) -
                              correction, 3) + (10.0 / ndim) * penalty(x)

    return result


@register_function('schaffer_f7_condition_10')
def schaffer_f7_condition_10(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    key1, key2 = jr.split(key)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=10.0)
    z = lamb @ Q @ tasy_func(R @ (x - x_opt))

    s = jnp.sqrt(z[:-1]**2 + z[:1]**2)

    term1 = (1 / ndim - 1) * jnp.sum(jnp.sqrt(s) + jnp.sqrt(s)
                                     * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2))

    result = jnp.power(term1, 2) + 10 * penalty(x)
    return result


@register_function('schaffer_f7_condition_1000')
def schaffer_f7_condition_1000(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    key1, key2 = jr.split(key)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=1000.0)
    z = lamb @ Q @ tasy_func(R @ (x - x_opt))

    s = jnp.sqrt(z[:-1]**2 + z[:1]**2)

    term1 = (1 / ndim - 1) * jnp.sum(jnp.sqrt(s) + jnp.sqrt(s)
                                     * jnp.power(jnp.sin(50.0 * jnp.power(s, 0.2)), 2))

    result = jnp.power(term1, 2) + 10 * penalty(x)
    return result


@register_function('griewank_rosenbrock_f8f2')
def griewank_rosenbrock_f8f2(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key)
    z = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0) * (R @ x) + 0.5
    s = 100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2

    return (10 / (ndim - 1)) * jnp.sum((s / 4000) - jnp.cos(s)) + 10.0


@register_function('schwefel_xsinx')
def schwefel_xsinx(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    ones = bernoulli_vector(ndim, key)
    lamb = lambda_func(ndim, alpha=10.0)
    x_hat = 2.0 * ones * x
    x_opt = 4.2096874633 / 2 * ones

    z_hat = x_hat.at[..., 1:].add(
        0.25 * (x_hat[..., :-1] - 2.0 * jnp.abs(x_opt[..., :-1])))

    z = 100.0 * (lamb @ (z_hat - 2.0 * jnp.abs(x_opt)) + 2.0 * jnp.abs(x_opt))

    f = -1.0 / (100.0 * ndim) * jnp.sum(z *
                                        jnp.sin(jnp.sqrt(jnp.abs(z))), axis=-1)

    # Penalization
    pen = 100.0 * penalty(z / 100.0)

    return f + 4.189828872724339 + pen


@register_function('gallagher_101_peaks')
def gallagher_101_peaks(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    key1, key2, key3, key4 = jr.split(key, 4)
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key1)

    w = 1.1 + 8.0 * (jnp.arange(1, 102, dtype=jnp.float32) - 2) / 99
    w = w.at[0].set(10.0)

    a = jnp.power(1000, 2.0 * (jnp.arange(100, dtype=jnp.float32) / 99.0))
    alpha = jr.permutation(key2, a)
    alpha = alpha.at[0].set(1000.0)

    C = jax.vmap(lambda_func, in_axes=(None, 0))(
        ndim, alpha)

    C /= jnp.power(alpha, 0.25)[:, None, None]

    y = jr.uniform(key3, shape=(101, ndim), minval=-5.0, maxval=5.0)
    y1 = jr.uniform(key4, shape=(ndim,), minval=-4.0, maxval=4.0)
    y = y.at[0].set(y1)

    diff = x[None, :] - y
    diff_rot = diff @ R.T

    def apply_C(C_i, d_i):
        return (d_i @ C_i) @ R.T

    diff_scaled = jax.vmap(apply_C)(C, diff_rot[1:])
    diff_scaled = jnp.vstack([diff_scaled[0:1], diff_scaled])

    in_bracket = -1.0 / (2.0 * ndim) * jnp.sum(diff_scaled * diff, axis=-1)

    inside_max = w * jnp.exp(in_bracket)
    f = 10.0 - jnp.max(inside_max)

    f_tosz = tosz_func(f)

    result = jnp.power(f_tosz, 2) + penalty(x)

    return result


@register_function('gallagher_21_peaks')
def gallagher_21_peaks(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    key1, key2, key3, key4 = jr.split(key, 4)

    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key1)

    w = 1.1 + 8.0 * (jnp.arange(1, 22, dtype=jnp.float32) - 2) / 19
    w = w.at[0].set(10.0)

    a = jnp.power(1000, 2.0 * (jnp.arange(20, dtype=jnp.float32) / 19.0))
    alpha = jr.permutation(key2, a)
    alpha = alpha.at[0].set(1000.0**2)

    C = jax.vmap(lambda_func, in_axes=(None, 0))(
        ndim, alpha)

    C /= jnp.power(alpha, 0.25)[:, None, None]

    y = jr.uniform(key3, shape=(21, ndim), minval=-4.9, maxval=4.9)
    y1 = jr.uniform(key4, shape=(ndim,), minval=-3.92, maxval=3.92)
    y = y.at[0].set(y1)

    diff = x[None, :] - y
    diff_rot = diff @ R.T

    def apply_C(C_i, d_i):
        return (d_i @ C_i) @ R.T

    diff_scaled = jax.vmap(apply_C)(C, diff_rot[1:])
    diff_scaled = jnp.vstack([diff_scaled[0:1], diff_scaled])

    in_bracket = -1.0 / (2.0 * ndim) * jnp.sum(diff_scaled * diff, axis=-1)

    inside_max = w * jnp.exp(in_bracket)
    f = 10.0 - jnp.max(inside_max)

    f_tosz = tosz_func(f)

    result = jnp.power(f_tosz, 2) + penalty(x)

    return result


@register_function('katsuura')
def katsuura(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    key1, key2 = jr.split(key)
    ndim = x.shape[-1]
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    lamb = lambda_func(ndim, alpha=100.0)
    z = Q @ lamb @ R @ (x - x_opt)

    J = 2.0 ** jnp.arange(1, 33, dtype=jnp.float32)  # (32,)
    # jsum term: shape (32, dim)
    z_expanded = z[None, :]  # (1, dim)
    J_expanded = J[:, None]  # (32, 1)
    jsum = jnp.abs(J_expanded * z_expanded -
                   jnp.round(J_expanded * z_expanded)) / J_expanded  # (32, dim)

    # Sum over j (the 32 terms)
    sum_j = jnp.sum(jsum, axis=0)  # (dim,)

    # Multiply by (1..dim) and add 1
    bracket = 1.0 + jnp.arange(1, ndim + 1, dtype=x.dtype) * sum_j  # (dim,)
    prod = jnp.prod(bracket)

    # Final scaling and power
    prod = jnp.power(prod, 10.0 / ndim ** 1.2)
    prod = prod * (10.0 / ndim ** 2.0) - (10.0 / ndim ** 2.0)

    return prod + penalty(x)


@register_function('lunacek_bi_rastrigin')
def lunacek_bi_rastrigin(x: jax.Array, key: PRNGKeyArray, x_opt: jax.Array) -> jax.Array:
    ndim = x.shape[-1]
    key1, key2, key3 = jr.split(key, 3)
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * jnp.sqrt(ndim + 20.0) - 8.2)

    x_opt = (mu0 / 2.0) * bernoulli_vector(ndim, key1)
    x_hat = 2 * jnp.sign(x_opt) * x

    R = rotation_matrix(ndim, key2)
    Q = rotation_matrix(ndim, key3)
    lamb = lambda_func(ndim, alpha=100.0)
    z = Q @ lamb @ R @ (x_hat - mu0 * jnp.ones_like(x))

    mu1 = -jnp.sqrt((mu0**2 - d) / s)

    term1 = jnp.minimum(jnp.sum(
        jnp.power(x_hat - mu0, 2)),
        d * ndim + s * jnp.sum(jnp.power(x_hat - mu1, 2))
    )

    term2 = 10.0 * (ndim - jnp.sum(jnp.cos(2.0 * jnp.pi * z)))

    return term1 + term2 + 1e4 * penalty(x)
