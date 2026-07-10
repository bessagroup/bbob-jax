"""CEC 2017 function characteristics metadata.

Maps each CEC 2017 function name to a dict with boolean flags
``unimodal``, ``multimodal``, ``hybrid``, ``composition``,
``rotated`` and ``structure_modified`` (see ``_cec2017_tags``
in ``spec.py`` for the schema; unlike CEC 2005 there is no
``noise`` key — the suite has no stochastic functions).
Derived from the :class:`~bbob_jax._src.spec.FunctionSpec`
table; a lookup with an unknown name raises ``KeyError``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import CEC2017_SPECS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

cec2017_function_characteristics: dict[str, dict[str, bool]] = {
    s.name: dict(s.tags) for s in CEC2017_SPECS
}
