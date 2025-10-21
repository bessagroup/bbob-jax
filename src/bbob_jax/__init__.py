"""
BBOB Benchmark set for Jax - BBOB Benchmark function implemented in JAX
"""

#                                                                       Modules
# =============================================================================

# Standard
from bbob_jax._src import bbob
from bbob_jax._src.plotting import plot_3d
from bbob_jax._src.registry import registry, registry_original

# Third-party

# Local

#                                                        Authorship and Credits
# =============================================================================
__author__ = 'Martin van der Schelling (m.p.vanderschelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
#
# =============================================================================


__all__ = [
    'plot_3d',
    'registry',
    'registry_original',
    'bbob',
]
