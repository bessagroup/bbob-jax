"""Registries for the BBOB, BBOB-noisy, CEC 2005 and CEC 2017 benchmarks.

The eight registry dicts are derived views of the
:class:`~bbob_jax._src.spec.FunctionSpec` table in ``spec.py``:
the randomized registries call each spec's maker as-is, the
``*_original`` registries bind ``deterministic=True`` (zero
shift, identity rotations, zero ``f_opt``). Each factory is
called as::

    fn, f_opt = registry["sphere"](ndim=2, key=jax_key)

where ``fn`` is a ``jax.tree_util.Partial`` with all
parameters bound and ``f_opt`` is the global minimum value.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

# Third-party
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray

# Local
from bbob_jax._src.factories import BBOBFn
from bbob_jax._src.spec import (
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

registry: dict[str, Callable[[int, PRNGKeyArray], BBOBFn]] = {
    s.name: s.maker for s in BBOB_SPECS
}

registry_original: dict[str, Callable[[int], BBOBFn]] = {
    s.name: Partial(s.maker, deterministic=True) for s in BBOB_SPECS
}

bbob_noisy_registry: dict[str, Callable] = {
    s.name: s.maker for s in BBOB_NOISY_SPECS
}

bbob_noisy_registry_original: dict[str, Callable] = {
    s.name: Partial(s.maker, deterministic=True) for s in BBOB_NOISY_SPECS
}

cec2005_registry: dict[str, Callable] = {
    s.name: s.maker for s in CEC2005_SPECS
}

cec2005_registry_original: dict[str, Callable] = {
    s.name: Partial(s.maker, deterministic=True) for s in CEC2005_SPECS
}

cec2017_registry: dict[str, Callable] = {
    s.name: s.maker for s in CEC2017_SPECS
}

cec2017_registry_original: dict[str, Callable] = {
    s.name: Partial(s.maker, deterministic=True) for s in CEC2017_SPECS
}

# CEC 2013 LSGO is a fixed-instance suite (parameters are official constants,
# not seed-sampled), so there is no ``*_original`` deterministic variant:
# the single registry IS the canonical instance. Each maker validates ndim
# against the function's native dimension and ignores ``key``.
cec2013lsgo_registry: dict[str, Callable] = {
    s.name: s.maker for s in CEC2013LSGO_SPECS
}
