"""CEC 2005 function characteristics metadata.

Maps each CEC 2005 function name to a dict with boolean
flags for unimodal, multimodal, composition, rotated,
noise, and structure_modified.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections import defaultdict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Schema keys:
#   unimodal, multimodal: mutually exclusive; composition implies multimodal
#   rotated: function applies a rotation matrix to the input
#   noise: function has stochastic noise; call signature is fn(x, key)
#          instead of fn(x)
#   structure_modified: function's mathematical structure is altered from
#                       the CEC 2005 spec for JAX compatibility

cec2005_function_characteristics: defaultdict = defaultdict(
    dict,
    {
        "f1": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f2": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f3": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f4": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise": True,
            "structure_modified": False,
        },
        "f5": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f6": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f7": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f8": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f9": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f10": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f11": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f12": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f13": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f14": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f15": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": False,
            "noise": False,
            "structure_modified": False,
        },
        "f16": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f17": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": True,
            "structure_modified": False,
        },
        "f18": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f19": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f20": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f21": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f22": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": False,
        },
        "f23": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": False,
            "structure_modified": True,
        },
        "f24": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": True,
            "structure_modified": True,
        },
        "f25": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise": True,
            "structure_modified": True,
        },
    },
)
