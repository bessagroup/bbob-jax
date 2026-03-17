#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

# Third-Party
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray

from bbob_jax._src.bbob import (
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
from bbob_jax._src.utils import fopt, rotation_matrix, xopt

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BBOBFn = tuple[Callable[[jax.Array], jax.Array], jax.Array]


def make_determinstic(
    fn: Callable, ndim: int, key: PRNGKeyArray | None = None
) -> BBOBFn:
    x_opt = jnp.zeros(ndim)
    eye = jnp.eye(ndim)
    f_opt = jnp.array(0.0)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=eye, Q=eye), f_opt


def make_randomized(fn: Callable, ndim: int, key: PRNGKeyArray) -> BBOBFn:
    key1, key2 = jr.split(key)
    x_opt = xopt(key1, ndim)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    f_opt = fopt(key)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=R, Q=Q), f_opt


registry: dict[str, Callable[[int, PRNGKeyArray], BBOBFn]] = {
    "attractive_sector": Partial(make_randomized, fn=attractive_sector),
    "bent_cigar": Partial(make_randomized, fn=bent_cigar),
    "discuss": Partial(make_randomized, fn=discuss),
    "ellipsoid": Partial(make_randomized, fn=ellipsoid),
    "ellipsoid_seperable": Partial(make_randomized, fn=ellipsoid_seperable),
    "gallagher_21_peaks": Partial(make_randomized, fn=gallagher_21_peaks),
    "gallagher_101_peaks": Partial(make_randomized, fn=gallagher_101_peaks),
    "griewank_rosenbrock_f8f2": Partial(
        make_randomized, fn=griewank_rosenbrock_f8f2
    ),
    "katsuura": Partial(make_randomized, fn=katsuura),
    "linear_slope": Partial(make_randomized, fn=linear_slope),
    "lunacek_bi_rastrigin": Partial(make_randomized, fn=lunacek_bi_rastrigin),
    "rastrigin": Partial(make_randomized, fn=rastrigin),
    "rastrigin_seperable": Partial(make_randomized, fn=rastrigin_seperable),
    "rosenbrock": Partial(make_randomized, fn=rosenbrock),
    "rosenbrock_rotated": Partial(make_randomized, fn=rosenbrock_rotated),
    "schaffer_f7_condition_10": Partial(
        make_randomized, fn=schaffer_f7_condition_10
    ),
    "schaffer_f7_condition_1000": Partial(
        make_randomized, fn=schaffer_f7_condition_1000
    ),
    "schwefel_xsinx": Partial(make_randomized, fn=schwefel_xsinx),
    "sharp_ridge": Partial(make_randomized, fn=sharp_ridge),
    "skew_rastrigin_bueche": Partial(
        make_randomized, fn=skew_rastrigin_bueche
    ),
    "sphere": Partial(make_randomized, fn=sphere),
    "step_ellipsoid": Partial(make_randomized, fn=step_ellipsoid),
    "sum_of_different_powers": Partial(
        make_randomized, fn=sum_of_different_powers
    ),
    "weierstrass": Partial(make_randomized, fn=weierstrass),
}

registry_original: dict[str, Callable[[int], BBOBFn]] = {
    "attractive_sector": Partial(make_determinstic, fn=attractive_sector),
    "bent_cigar": Partial(make_determinstic, fn=bent_cigar),
    "discuss": Partial(make_determinstic, fn=discuss),
    "ellipsoid": Partial(make_determinstic, fn=ellipsoid),
    "ellipsoid_seperable": Partial(make_determinstic, fn=ellipsoid_seperable),
    "gallagher_21_peaks": Partial(make_determinstic, fn=gallagher_21_peaks),
    "gallagher_101_peaks": Partial(make_determinstic, fn=gallagher_101_peaks),
    "griewank_rosenbrock_f8f2": Partial(
        make_determinstic, fn=griewank_rosenbrock_f8f2
    ),
    "katsuura": Partial(make_determinstic, fn=katsuura),
    "linear_slope": Partial(make_determinstic, fn=linear_slope),
    "lunacek_bi_rastrigin": Partial(
        make_determinstic, fn=lunacek_bi_rastrigin
    ),
    "rastrigin": Partial(make_determinstic, fn=rastrigin),
    "rastrigin_seperable": Partial(make_determinstic, fn=rastrigin_seperable),
    "rosenbrock": Partial(make_determinstic, fn=rosenbrock),
    "rosenbrock_rotated": Partial(make_determinstic, fn=rosenbrock_rotated),
    "schaffer_f7_condition_10": Partial(
        make_determinstic, fn=schaffer_f7_condition_10
    ),
    "schaffer_f7_condition_1000": Partial(
        make_determinstic, fn=schaffer_f7_condition_1000
    ),
    "schwefel_xsinx": Partial(make_determinstic, fn=schwefel_xsinx),
    "sharp_ridge": Partial(make_determinstic, fn=sharp_ridge),
    "skew_rastrigin_bueche": Partial(
        make_determinstic, fn=skew_rastrigin_bueche
    ),
    "sphere": Partial(make_determinstic, fn=sphere),
    "step_ellipsoid": Partial(make_determinstic, fn=step_ellipsoid),
    "sum_of_different_powers": Partial(
        make_determinstic, fn=sum_of_different_powers
    ),
    "weierstrass": Partial(make_determinstic, fn=weierstrass),
}


def make_randomized_cec2005(
    fn: Callable, ndim: int, key: PRNGKeyArray, num_components: int = 1
) -> BBOBFn:
    """Factory for CEC 2005 functions with seed-generated parameters.

    Always splits key into 2*num_components+2 subkeys so x_opt, R, Q,
    and f_opt consume distinct subkeys (avoids the BBOB key-reuse pattern).
    """
    total_keys = 2 * num_components + 2
    keys = jr.split(key, total_keys)
    # keys[0:num_components]              → R matrices
    # keys[num_components:2*num_components] → Q matrices
    # keys[-2]                            → x_opt seed
    # keys[-1]                            → f_opt seed

    if num_components == 1:
        x_opt = jr.uniform(
            keys[-2], shape=(ndim,), minval=-100.0, maxval=100.0
        )
        R = rotation_matrix(ndim, keys[0])
        Q = rotation_matrix(ndim, keys[num_components])
    else:
        xopt_keys = jr.split(keys[-2], num_components)
        x_opt = jnp.stack(
            [
                jr.uniform(
                    xopt_keys[i], shape=(ndim,), minval=-100.0, maxval=100.0
                )
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


def make_deterministic_cec2005(
    fn: Callable,
    ndim: int,
    key: PRNGKeyArray | None = None,
    num_components: int = 1,
) -> BBOBFn:
    """Factory for CEC 2005 functions with zero shift and identity rotations.

    key is accepted and ignored so both registries have identical signatures.
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


_NC = 10  # num_components for all composition functions (F15-F25)

cec2005_registry: dict[str, Callable] = {
    "f1": Partial(make_randomized_cec2005, fn=f1, num_components=1),
    "f2": Partial(make_randomized_cec2005, fn=f2, num_components=1),
    "f3": Partial(make_randomized_cec2005, fn=f3, num_components=1),
    "f4": Partial(make_randomized_cec2005, fn=f4, num_components=1),
    "f5": Partial(make_randomized_cec2005, fn=f5, num_components=1),
    "f6": Partial(make_randomized_cec2005, fn=f6, num_components=1),
    "f7": Partial(make_randomized_cec2005, fn=f7, num_components=1),
    "f8": Partial(make_randomized_cec2005, fn=f8, num_components=1),
    "f9": Partial(make_randomized_cec2005, fn=f9, num_components=1),
    "f10": Partial(make_randomized_cec2005, fn=f10, num_components=1),
    "f11": Partial(make_randomized_cec2005, fn=f11, num_components=1),
    "f12": Partial(make_randomized_cec2005, fn=f12, num_components=1),
    "f13": Partial(make_randomized_cec2005, fn=f13, num_components=1),
    "f14": Partial(make_randomized_cec2005, fn=f14, num_components=1),
    "f15": Partial(make_randomized_cec2005, fn=f15, num_components=_NC),
    "f16": Partial(make_randomized_cec2005, fn=f16, num_components=_NC),
    "f17": Partial(make_randomized_cec2005, fn=f17, num_components=_NC),
    "f18": Partial(make_randomized_cec2005, fn=f18, num_components=_NC),
    "f19": Partial(make_randomized_cec2005, fn=f19, num_components=_NC),
    "f20": Partial(make_randomized_cec2005, fn=f20, num_components=_NC),
    "f21": Partial(make_randomized_cec2005, fn=f21, num_components=_NC),
    "f22": Partial(make_randomized_cec2005, fn=f22, num_components=_NC),
    "f23": Partial(make_randomized_cec2005, fn=f23, num_components=_NC),
    "f24": Partial(make_randomized_cec2005, fn=f24, num_components=_NC),
    "f25": Partial(make_randomized_cec2005, fn=f25, num_components=_NC),
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
    "f15": Partial(make_deterministic_cec2005, fn=f15, num_components=_NC),
    "f16": Partial(make_deterministic_cec2005, fn=f16, num_components=_NC),
    "f17": Partial(make_deterministic_cec2005, fn=f17, num_components=_NC),
    "f18": Partial(make_deterministic_cec2005, fn=f18, num_components=_NC),
    "f19": Partial(make_deterministic_cec2005, fn=f19, num_components=_NC),
    "f20": Partial(make_deterministic_cec2005, fn=f20, num_components=_NC),
    "f21": Partial(make_deterministic_cec2005, fn=f21, num_components=_NC),
    "f22": Partial(make_deterministic_cec2005, fn=f22, num_components=_NC),
    "f23": Partial(make_deterministic_cec2005, fn=f23, num_components=_NC),
    "f24": Partial(make_deterministic_cec2005, fn=f24, num_components=_NC),
    "f25": Partial(make_deterministic_cec2005, fn=f25, num_components=_NC),
}
