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
from bbob_jax._src.utils import fopt, rotation_matrix, xopt

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BBOBFn = Callable[[jax.Array, PRNGKeyArray], jax.Array]


def make_determinstic(
    fn: Callable, ndim: int, key: PRNGKeyArray | None = None
) -> BBOBFn:
    x_opt = jnp.zeros(ndim)
    eye = jnp.eye(ndim)
    f_opt = jnp.array(0.0)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=eye, Q=eye)


def make_randomized(fn: Callable, ndim: int, key: PRNGKeyArray) -> BBOBFn:
    key1, key2 = jr.split(key)
    x_opt = xopt(key1, ndim)
    R = rotation_matrix(ndim, key1)
    Q = rotation_matrix(ndim, key2)
    f_opt = fopt(key)
    return Partial(fn, x_opt=x_opt, f_opt=f_opt, R=R, Q=Q)


registry: dict[str, Callable[[Callable, int, PRNGKeyArray], BBOBFn]] = {
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

registry_original: dict[str, Callable[[Callable, int], BBOBFn]] = {
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
