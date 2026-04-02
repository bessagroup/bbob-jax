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
#   noise_omitted: function is noisy per CEC 2005 spec but noise is omitted
#                  here for jax.grad compatibility
#   grad_safe_approximation: function retains a smooth approximation instead
#                            of the paper's exact branching/non-continuity
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
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f2": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f3": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f4": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise_omitted": True,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f5": {
            "unimodal": True,
            "multimodal": False,
            "composition": False,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f6": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f7": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f8": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f9": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f10": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f11": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f12": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f13": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f14": {
            "unimodal": False,
            "multimodal": True,
            "composition": False,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": False,
            "structure_modified": False,
        },
        "f15": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": False,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f16": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f17": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": True,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f18": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f19": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f20": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f21": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f22": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": False,
        },
        "f23": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": False,
            "grad_safe_approximation": True,
            "structure_modified": True,
        },
        "f24": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": True,
            "grad_safe_approximation": True,
            "structure_modified": True,
        },
        "f25": {
            "unimodal": False,
            "multimodal": True,
            "composition": True,
            "rotated": True,
            "noise_omitted": True,
            "grad_safe_approximation": True,
            "structure_modified": True,
        },
    },
)
