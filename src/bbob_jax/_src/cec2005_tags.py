"""CEC 2005 function characteristics and deviation metadata.

``cec2005_function_characteristics`` maps each CEC 2005 function name
to a dict with boolean flags ``unimodal``, ``multimodal``,
``composition``, ``rotated``, ``noise`` and ``structure_modified``
(see ``_cec_tags`` in ``spec.py`` for the schema).

``cec2005_deviations`` maps each function name to a tuple of
plain-language sentences, one per way the bbob-jax function departs
from the official CEC 2005 definition. Every tuple starts with the
suite-wide entries (generated instance parameters and ``f_opt``);
F23-F25 add their soft rounding, and F24/F25 their noise-free height
normalization. The composition weights and noise of F15-F25, and the
other functions' height normalization, follow the official C code and
have no entries. It is the source for a table of deviations.

Both are derived from the :class:`~bbob_jax._src.spec.FunctionSpec`
table; a lookup with an unknown name raises ``KeyError``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import CEC2005_SPECS, CEC2005_SUITE_DEVIATIONS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

cec2005_function_characteristics: dict[str, dict[str, bool]] = {
    s.name: dict(s.tags) for s in CEC2005_SPECS
}

cec2005_deviations: dict[str, tuple[str, ...]] = {
    s.name: CEC2005_SUITE_DEVIATIONS + s.deviations for s in CEC2005_SPECS
}
"""How each CEC 2005 function departs from the official definition.

Maps the function name (``"f1"`` ... ``"f25"``) to a tuple of
plain-language sentences. Every tuple starts with the suite-wide
entries; F23-F25 add their soft rounding and F24/F25 their noise-free
normalization. Example, one row per deviation::

    for name, entries in cec2005_deviations.items():
        for sentence in entries:
            print(name, sentence)
"""
