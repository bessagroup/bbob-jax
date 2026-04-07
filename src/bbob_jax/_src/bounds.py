"""Search-space bounds for BBOB and CEC 2005 benchmarks."""

#                                                                       Modules
# =============================================================================

# Standard
import math

# Local
from bbob_jax._src.registry import registry

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BBOB_BOUNDS: tuple[float, float] = (-5.0, 5.0)

bbob_bounds: dict[str, tuple[float, float]] = {
    name: BBOB_BOUNDS for name in registry
}

cec2005_bounds: dict[str, tuple[float, float]] = {
    "f1": (-100.0, 100.0),
    "f2": (-100.0, 100.0),
    "f3": (-100.0, 100.0),
    "f4": (-100.0, 100.0),
    "f5": (-100.0, 100.0),
    "f6": (-100.0, 100.0),
    "f7": (0.0, 600.0),
    "f8": (-32.0, 32.0),
    "f9": (-5.0, 5.0),
    "f10": (-5.0, 5.0),
    "f11": (-0.5, 0.5),
    "f12": (-math.pi, math.pi),
    "f13": (-3.0, 1.0),
    "f14": (-100.0, 100.0),
    "f15": (-5.0, 5.0),
    "f16": (-5.0, 5.0),
    "f17": (-5.0, 5.0),
    "f18": (-5.0, 5.0),
    "f19": (-5.0, 5.0),
    "f20": (-5.0, 5.0),
    "f21": (-5.0, 5.0),
    "f22": (-5.0, 5.0),
    "f23": (-5.0, 5.0),
    "f24": (-5.0, 5.0),
    "f25": (2.0, 5.0),
}
