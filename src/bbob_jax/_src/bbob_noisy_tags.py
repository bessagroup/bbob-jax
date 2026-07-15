"""BBOB-noisy function characteristics metadata.

Maps each BBOB-noisy function name to a dict with boolean flags
``separable``, ``unimodal``, ``gaussian_noise``,
``uniform_noise``, ``cauchy_noise``, ``severe`` and ``noise``.
``separable``/``unimodal`` describe the undisturbed base
function. Derived from the
:class:`~bbob_jax._src.spec.FunctionSpec` table; a lookup with
an unknown name raises ``KeyError``.
"""

#                                                                       Modules
# =============================================================================

# Local
from bbob_jax._src.spec import BBOB_NOISY_SPECS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

bbob_noisy_function_characteristics: dict[str, dict[str, bool]] = {
    s.name: dict(s.tags) for s in BBOB_NOISY_SPECS
}
