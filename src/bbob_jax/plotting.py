"""
Module for plotting BBob functions in 2D and 3D.
"""

from bbob_jax._src.plotting import _create_mesh, plot_2d, plot_3d

#                                                        Authorship and Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
#
# =============================================================================

__all__ = [
    "plot_2d",
    "plot_3d",
    "_create_mesh",
]
