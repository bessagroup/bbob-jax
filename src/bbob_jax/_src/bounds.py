"""Search-space bounds for the BBOB, BBOB-noisy, CEC 2005 and
CEC 2017 benchmarks.

Derived from the :class:`~bbob_jax._src.spec.FunctionSpec`
table in ``spec.py``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import (
    BBOB_BOUNDS,
    BBOB_NOISY_SPECS,
    BBOB_SPECS,
    CEC2005_SPECS,
    CEC2013LSGO_SPECS,
    CEC2017_SPECS,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "BBOB_BOUNDS",
    "bbob_bounds",
    "bbob_noisy_bounds",
    "cec2005_bounds",
    "cec2017_bounds",
    "cec2013lsgo_bounds",
]

bbob_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in BBOB_SPECS
}

bbob_noisy_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in BBOB_NOISY_SPECS
}

cec2005_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in CEC2005_SPECS
}

cec2017_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in CEC2017_SPECS
}

cec2013lsgo_bounds: dict[str, tuple[float, float]] = {
    s.name: s.bounds for s in CEC2013LSGO_SPECS
}
