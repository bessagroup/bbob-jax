#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.registry import cec2005_registry, registry

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BBOB_BOUNDS: tuple[float, float] = (-5.0, 5.0)
CEC2005_BOUNDS: tuple[float, float] = (-100.0, 100.0)

bbob_bounds: dict[str, tuple[float, float]] = {
    name: BBOB_BOUNDS for name in registry
}

cec2005_bounds: dict[str, tuple[float, float]] = {
    name: CEC2005_BOUNDS for name in cec2005_registry
}
