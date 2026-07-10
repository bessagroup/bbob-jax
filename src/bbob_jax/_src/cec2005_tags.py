"""CEC 2005 function characteristics metadata.

Maps each CEC 2005 function name to a dict with boolean flags
``unimodal``, ``multimodal``, ``composition``, ``rotated``,
``noise`` and ``structure_modified`` (see ``_cec_tags`` in
``spec.py`` for the schema). Derived from the
:class:`~bbob_jax._src.spec.FunctionSpec` table; a lookup with
an unknown name raises ``KeyError``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import CEC2005_SPECS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

cec2005_function_characteristics: dict[str, dict[str, bool]] = {
    s.name: dict(s.tags) for s in CEC2005_SPECS
}
