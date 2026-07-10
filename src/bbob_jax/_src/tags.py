"""BBOB function characteristics metadata.

Maps each BBOB function name to a dict with boolean flags
``separable`` and ``unimodal``. Derived from the
:class:`~bbob_jax._src.spec.FunctionSpec` table; a lookup with
an unknown name raises ``KeyError``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import BBOB_SPECS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

function_characteristics: dict[str, dict[str, bool]] = {
    s.name: dict(s.tags) for s in BBOB_SPECS
}
