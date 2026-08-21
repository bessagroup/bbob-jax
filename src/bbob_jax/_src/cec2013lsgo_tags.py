"""CEC 2013 LSGO function characteristics metadata.

Maps each CEC 2013 LSGO function name to a dict with the category flags
``separable``, ``partially_separable``, ``overlapping`` and
``non_separable`` (exactly one True, following the Li et al. 2013 groups)
plus ``rotated`` (True for the functions carrying subcomponent rotation
matrices). See ``_cec2013lsgo_tags`` in ``spec.py`` for the schema. Derived
from the :class:`~bbob_jax._src.spec.FunctionSpec` table; a lookup with an
unknown name raises ``KeyError``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import CEC2013LSGO_SPECS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

cec2013lsgo_function_characteristics: dict[str, dict[str, bool]] = {
    s.name: dict(s.tags) for s in CEC2013LSGO_SPECS
}
