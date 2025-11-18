"""
BBOB Benchmark set for Jax - BBOB Benchmark function implemented in JAX
"""

#                                                                       Modules
# =============================================================================

# Standard
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
from bbob_jax._src.registry import registry, registry_original
from bbob_jax._src.tags import function_characteristics

#                                                        Authorship and Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
#
# =============================================================================


__all__ = [
    "registry",
    "registry_original",
    "function_characteristics",
    "bbob",
    "attractive_sector",
    "bent_cigar",
    "discuss",
    "ellipsoid",
    "ellipsoid_seperable",
    "gallagher_21_peaks",
    "gallagher_101_peaks",
    "griewank_rosenbrock_f8f2",
    "katsuura",
    "linear_slope",
    "lunacek_bi_rastrigin",
    "rastrigin",
    "rastrigin_seperable",
    "rosenbrock",
    "rosenbrock_rotated",
    "schaffer_f7_condition_10",
    "schaffer_f7_condition_1000",
    "schwefel_xsinx",
    "sharp_ridge",
    "skew_rastrigin_bueche",
    "sphere",
    "step_ellipsoid",
    "sum_of_different_powers",
    "weierstrass",
]
