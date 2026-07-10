"""Search-space bounds for BBOB and CEC 2005 benchmarks.

Derived from the :class:`~bbob_jax._src.spec.FunctionSpec`
table in ``spec.py``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import BBOB_BOUNDS, BBOB_SPECS, CEC2005_SPECS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = ["BBOB_BOUNDS", "bbob_bounds", "cec2005_bounds"]

bbob_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in BBOB_SPECS
}

cec2005_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in CEC2005_SPECS
}
