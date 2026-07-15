"""Single source of truth for every benchmark function.

Each :class:`FunctionSpec` row bundles everything the package
knows about one function: its implementation, the factory that
constructs problem instances (randomized or deterministic),
its metadata tags, its search-space bounds and where its true
optimum lives. The public registries, tag dicts and bounds
dicts in ``registry.py``, ``tags.py``, ``cec2005_tags.py``,
``cec2017_tags.py`` and ``bounds.py`` are all derived views of
this table — adding a function means adding its implementation
and one row here.
"""

#                                                                       Modules
# =============================================================================

# Standard
import math
from collections.abc import Callable
from typing import Any, NamedTuple, cast

# Third-party
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from bbob_jax._src import bbob_noisy, cec2017

# Local
from bbob_jax._src.bbob import (
    attractive_sector,
    bent_cigar,
    discuss,
    ellipsoid,
    ellipsoid_seperable,
    gallagher_21_peaks,
    gallagher_101_peaks,
    griewank_rosenbrock_f8f2,
    katsuura,
    linear_slope,
    lunacek_bi_rastrigin,
    rastrigin,
    rastrigin_seperable,
    rosenbrock,
    rosenbrock_rotated,
    schaffer_f7_condition_10,
    schaffer_f7_condition_1000,
    schwefel_xsinx,
    sharp_ridge,
    skew_rastrigin_bueche,
    sphere,
    step_ellipsoid,
    sum_of_different_powers,
    weierstrass,
)
from bbob_jax._src.cec2005 import (
    _composition1_fns,
    _composition2_fns,
    _composition3_fns,
    f1,
    f2,
    f3,
    f4,
    f4_true,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f24_true,
    f25,
)
from bbob_jax._src.factories import (
    BBOBFn,
    _add_f_max,
    _make_bueche,
    _make_cec2005_conditioned_single,
    _make_cec2005_f5,
    _make_cec2005_f7,
    _make_cec2005_f8,
    _make_cec2005_f12,
    _make_cec2005_f18_family,
    _make_cec2005_f20,
    _make_gallagher,
    _make_linear_slope,
    _make_lunacek,
    _make_schwefel,
    _make_with_mat,
    make_bbob,
    make_cec2005,
    make_cec2005_conditioned,
    make_cec2017,
    make_cec2017_composition,
    make_cec2017_hybrid,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BBOB_BOUNDS: tuple[float, float] = (-5.0, 5.0)


#                                                              x_opt resolvers
# =============================================================================
# Given the keyword arguments bound into a constructed Partial, return the
# input location of the function's global minimum. For most functions this
# is the sampled ``x_opt``; the resolvers below encode the exceptions.


def _default_x_opt(kw: dict[str, Any], ndim: int) -> jax.Array:
    """Minimum is at the sampled ``x_opt``."""
    return cast(jax.Array, kw["x_opt"])


def _first_component_x_opt(kw: dict[str, Any], ndim: int) -> jax.Array:
    """Compositions: the global minimum is the first component's optimum."""
    return cast(jax.Array, kw["x_opt"][0])


def _linear_slope_x_opt(kw: dict[str, Any], ndim: int) -> jax.Array:
    """linear_slope: the minimum is at the boundary point ``_ls_x_opt``."""
    return cast(jax.Array, kw["_ls_x_opt"])


def _rosenbrock_rotated_x_opt(kw: dict[str, Any], ndim: int) -> jax.Array:
    """rosenbrock_rotated: minimum offset from ``x_opt`` (see docstring).

    ``z = zmax * ((x - x_opt) @ R) + 0.5`` is 1 at the minimum, so the
    minimizer is ``x_opt + (0.5 / zmax) * ones @ R.T``. It can fall
    outside the standard bounds for some rotations.
    """
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
    return cast(
        jax.Array, kw["x_opt"] + (0.5 / zmax) * (jnp.ones(ndim) @ kw["R"].T)
    )


def _griewank_rosenbrock_x_opt(kw: dict[str, Any], ndim: int) -> jax.Array:
    """griewank_rosenbrock_f8f2: minimum offset from ``x_opt``.

    ``z = zmax * (R @ (x - x_opt)) + 0.5`` is 1 at the minimum, so the
    minimizer is ``x_opt + (0.5 / zmax) * R.T @ ones``. It can fall
    outside the standard bounds for some rotations.
    """
    zmax = jnp.maximum(1.0, jnp.sqrt(ndim) / 8.0)
    return cast(
        jax.Array, kw["x_opt"] + (0.5 / zmax) * (kw["R"].T @ jnp.ones(ndim))
    )


def _cec2017_levy_x_opt(kw: dict[str, Any], ndim: int) -> jax.Array:
    """cec2017_f9: the Levy kernel's minimum is at ``z = ones``.

    ``z = R @ (x - x_opt)`` is all-ones at the minimum, so the
    minimizer is ``x_opt + R.T @ ones`` — not the sampled shift,
    matching the official reference code (see ``cec2017.f9``).
    """
    return cast(jax.Array, kw["x_opt"] + kw["R"].T @ jnp.ones(ndim))


#                                                                  Tag schemas
# =============================================================================


def _bbob_tags(*, separable: bool, unimodal: bool) -> dict[str, bool]:
    """BBOB tag schema: ``separable`` and ``unimodal``."""
    return {"separable": separable, "unimodal": unimodal}


def _bbob_noisy_tags(
    *, separable: bool, unimodal: bool, model: str, severe: bool
) -> dict[str, bool]:
    """BBOB-noisy tag schema.

    ``separable`` and ``unimodal`` describe the undisturbed base
    function, mirroring the noiseless suite's labels. Exactly one
    of the three noise-model flags is True. ``severe`` is False
    for the moderate-noise group f101-f106. ``noise`` is True on
    every row: the call signature is ``fn(x, key)``.
    """
    if model not in ("gauss", "uniform", "cauchy"):
        raise ValueError(f"Unknown noise model: {model}")
    return {
        "separable": separable,
        "unimodal": unimodal,
        "gaussian_noise": model == "gauss",
        "uniform_noise": model == "uniform",
        "cauchy_noise": model == "cauchy",
        "severe": severe,
        "noise": True,
    }


def _cec_tags(
    *,
    unimodal: bool = False,
    composition: bool = False,
    rotated: bool = False,
    noise: bool = False,
    structure_modified: bool = False,
) -> dict[str, bool]:
    """CEC 2005 tag schema.

    ``unimodal`` and ``multimodal`` are mutually exclusive;
    ``composition`` implies multimodal. ``noise`` marks the
    stochastic functions whose call signature is ``fn(x, key)``
    instead of ``fn(x)``. ``structure_modified`` marks functions
    whose mathematical structure deviates from the CEC 2005 spec
    for JAX compatibility.
    """
    return {
        "unimodal": unimodal,
        "multimodal": not unimodal,
        "composition": composition,
        "rotated": rotated,
        "noise": noise,
        "structure_modified": structure_modified,
    }


def _cec2017_tags(
    *,
    unimodal: bool = False,
    hybrid: bool = False,
    composition: bool = False,
    rotated: bool = True,
    structure_modified: bool = False,
) -> dict[str, bool]:
    """CEC 2017 tag schema.

    ``unimodal`` and ``multimodal`` are mutually exclusive;
    ``hybrid`` marks F11-F20 (shuffled dimension chunks),
    ``composition`` marks F21-F30; both imply multimodal.
    There is no ``noise`` key — the suite has no stochastic
    functions. ``rotated`` defaults to True (every function is
    shifted and rotated except F6, whose rotation is dead code
    in the official reference — see ``cec2017.f6``).
    ``structure_modified`` marks deviations from the reference
    code for JAX compatibility.
    """
    return {
        "unimodal": unimodal,
        "multimodal": not unimodal,
        "hybrid": hybrid,
        "composition": composition,
        "rotated": rotated,
        "structure_modified": structure_modified,
    }


#                                                                 FunctionSpec
# =============================================================================


class FunctionSpec(NamedTuple):
    """Everything the package knows about one benchmark function.

    Attributes
    ----------
    name : str
        Registry key of the function.
    suite : str
        Benchmark suite, ``"bbob"``, ``"bbob_noisy"``,
        ``"cec2005"`` or ``"cec2017"``.
    maker : Callable
        Factory constructing a problem instance. Called as
        ``maker(ndim=..., key=..., deterministic=...)`` and
        returning ``(fn, f_opt)``.
    tags : dict[str, bool]
        Function characteristics (suite-specific schema).
    bounds : tuple[float, float]
        Search-space box bounds.
    x_opt_from : Callable
        Resolver mapping the constructed Partial's keyword dict
        (plus ``ndim``) to the location of the global minimum.
    min_ndim : int
        Smallest ``ndim`` the function is defined for. Makers
        raise ``ValueError`` below it (e.g. CEC 2017 hybrids
        need one dimension per subcomponent kernel).
    true_fn : Callable or None
        For noisy functions, the undisturbed implementation with
        the same bound-parameter signature minus ``key``;
        ``problem()`` binds it as ``Problem.fn_true``. None for
        noise-free functions (``fn_true`` is then ``fn`` itself).
    """

    name: str
    suite: str
    maker: Callable[..., BBOBFn]
    tags: dict[str, bool]
    bounds: tuple[float, float]
    x_opt_from: Callable[[dict[str, Any], int], jax.Array] = _default_x_opt
    min_ndim: int = 1
    true_fn: Callable[..., jax.Array] | None = None


#                                                                   BBOB table
# =============================================================================

BBOB_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        name="attractive_sector",
        suite="bbob",
        maker=Partial(
            _make_with_mat, fn=attractive_sector, alpha=10.0, order="QLR"
        ),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="bent_cigar",
        suite="bbob",
        maker=Partial(make_bbob, fn=bent_cigar),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="discuss",
        suite="bbob",
        maker=Partial(make_bbob, fn=discuss),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="ellipsoid",
        suite="bbob",
        maker=Partial(make_bbob, fn=ellipsoid),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="ellipsoid_seperable",
        suite="bbob",
        maker=Partial(make_bbob, fn=ellipsoid_seperable),
        tags=_bbob_tags(separable=True, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="gallagher_21_peaks",
        suite="bbob",
        maker=Partial(
            _make_gallagher,
            fn=gallagher_21_peaks,
            num_peaks=21,
            w_divisor=19,
            alpha_first=1000.0**2,
            y_minval=-4.9,
            y_maxval=4.9,
        ),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="gallagher_101_peaks",
        suite="bbob",
        maker=Partial(
            _make_gallagher,
            fn=gallagher_101_peaks,
            num_peaks=101,
            w_divisor=99,
            alpha_first=1000.0,
            y_minval=-5.0,
            y_maxval=5.0,
        ),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="griewank_rosenbrock_f8f2",
        suite="bbob",
        maker=Partial(make_bbob, fn=griewank_rosenbrock_f8f2),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
        x_opt_from=_griewank_rosenbrock_x_opt,
    ),
    FunctionSpec(
        name="katsuura",
        suite="bbob",
        maker=Partial(_make_with_mat, fn=katsuura, alpha=100.0, order="QLR"),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="linear_slope",
        suite="bbob",
        maker=Partial(_make_linear_slope, fn=linear_slope),
        tags=_bbob_tags(separable=True, unimodal=True),
        bounds=BBOB_BOUNDS,
        x_opt_from=_linear_slope_x_opt,
    ),
    FunctionSpec(
        name="lunacek_bi_rastrigin",
        suite="bbob",
        maker=Partial(_make_lunacek, fn=lunacek_bi_rastrigin),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="rastrigin",
        suite="bbob",
        maker=Partial(_make_with_mat, fn=rastrigin, alpha=10.0, order="RLQ"),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    # NOTE: rastrigin_seperable and skew_rastrigin_bueche were tagged
    # unimodal=True before the spec table existed; both are Rastrigin
    # variants and highly multimodal.
    FunctionSpec(
        name="rastrigin_seperable",
        suite="bbob",
        maker=Partial(make_bbob, fn=rastrigin_seperable),
        tags=_bbob_tags(separable=True, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="rosenbrock",
        suite="bbob",
        maker=Partial(make_bbob, fn=rosenbrock),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="rosenbrock_rotated",
        suite="bbob",
        maker=Partial(make_bbob, fn=rosenbrock_rotated),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
        x_opt_from=_rosenbrock_rotated_x_opt,
    ),
    FunctionSpec(
        name="schaffer_f7_condition_10",
        suite="bbob",
        maker=Partial(
            _make_with_mat,
            fn=schaffer_f7_condition_10,
            alpha=10.0,
            order="LQ",
        ),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="schaffer_f7_condition_1000",
        suite="bbob",
        maker=Partial(
            _make_with_mat,
            fn=schaffer_f7_condition_1000,
            alpha=1000.0,
            order="LQ",
        ),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="schwefel_xsinx",
        suite="bbob",
        maker=Partial(_make_schwefel, fn=schwefel_xsinx),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="sharp_ridge",
        suite="bbob",
        maker=Partial(_make_with_mat, fn=sharp_ridge, alpha=10.0, order="QLR"),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="skew_rastrigin_bueche",
        suite="bbob",
        maker=Partial(_make_bueche, fn=skew_rastrigin_bueche),
        tags=_bbob_tags(separable=True, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="sphere",
        suite="bbob",
        maker=Partial(make_bbob, fn=sphere),
        tags=_bbob_tags(separable=True, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="step_ellipsoid",
        suite="bbob",
        maker=Partial(
            _make_with_mat, fn=step_ellipsoid, alpha=10.0, order="LR"
        ),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="sum_of_different_powers",
        suite="bbob",
        maker=Partial(make_bbob, fn=sum_of_different_powers),
        tags=_bbob_tags(separable=False, unimodal=True),
        bounds=BBOB_BOUNDS,
    ),
    FunctionSpec(
        name="weierstrass",
        suite="bbob",
        maker=Partial(_make_with_mat, fn=weierstrass, alpha=0.01, order="RLQ"),
        tags=_bbob_tags(separable=False, unimodal=False),
        bounds=BBOB_BOUNDS,
    ),
)

#                                                             BBOB-noisy table
# =============================================================================
# f101-f130: eight base landscapes x three noise models (Gaussian, uniform,
# Cauchy — always in that order within a triple), moderate severity for
# f101-f106 and severe for f107-f130. ``true_fn`` is the shared undisturbed
# base of each triple.

BBOB_NOISY_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        name="bbob_noisy_f101",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f101),
        tags=_bbob_noisy_tags(
            separable=True, unimodal=True, model="gauss", severe=False
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.sphere_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f102",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f102),
        tags=_bbob_noisy_tags(
            separable=True, unimodal=True, model="uniform", severe=False
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.sphere_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f103",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f103),
        tags=_bbob_noisy_tags(
            separable=True, unimodal=True, model="cauchy", severe=False
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.sphere_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f104",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f104),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="gauss", severe=False
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f105",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f105),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="uniform", severe=False
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f106",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f106),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="cauchy", severe=False
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f107",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f107),
        tags=_bbob_noisy_tags(
            separable=True, unimodal=True, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.sphere_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f108",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f108),
        tags=_bbob_noisy_tags(
            separable=True, unimodal=True, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.sphere_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f109",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f109),
        tags=_bbob_noisy_tags(
            separable=True, unimodal=True, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.sphere_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f110",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f110),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f111",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f111),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f112",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f112),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f113",
        suite="bbob_noisy",
        maker=Partial(
            _make_with_mat, fn=bbob_noisy.f113, alpha=10.0, order="LR"
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.step_ellipsoid_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f114",
        suite="bbob_noisy",
        maker=Partial(
            _make_with_mat, fn=bbob_noisy.f114, alpha=10.0, order="LR"
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.step_ellipsoid_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f115",
        suite="bbob_noisy",
        maker=Partial(
            _make_with_mat, fn=bbob_noisy.f115, alpha=10.0, order="LR"
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.step_ellipsoid_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f116",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f116),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.ellipsoid_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f117",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f117),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.ellipsoid_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f118",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f118),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.ellipsoid_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f119",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f119),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.different_powers_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f120",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f120),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.different_powers_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f121",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f121),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=True, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.different_powers_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f122",
        suite="bbob_noisy",
        maker=Partial(
            _make_with_mat, fn=bbob_noisy.f122, alpha=10.0, order="LQ"
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.schaffer_f7_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f123",
        suite="bbob_noisy",
        maker=Partial(
            _make_with_mat, fn=bbob_noisy.f123, alpha=10.0, order="LQ"
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.schaffer_f7_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f124",
        suite="bbob_noisy",
        maker=Partial(
            _make_with_mat, fn=bbob_noisy.f124, alpha=10.0, order="LQ"
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.schaffer_f7_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f125",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f125),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        x_opt_from=_griewank_rosenbrock_x_opt,
        true_fn=bbob_noisy.griewank_rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f126",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f126),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        x_opt_from=_griewank_rosenbrock_x_opt,
        true_fn=bbob_noisy.griewank_rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f127",
        suite="bbob_noisy",
        maker=Partial(make_bbob, fn=bbob_noisy.f127),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        x_opt_from=_griewank_rosenbrock_x_opt,
        true_fn=bbob_noisy.griewank_rosenbrock_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f128",
        suite="bbob_noisy",
        maker=Partial(
            _make_gallagher,
            fn=bbob_noisy.f128,
            num_peaks=101,
            w_divisor=99,
            alpha_first=1000.0,
            y_minval=-5.0,
            y_maxval=5.0,
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="gauss", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.gallagher_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f129",
        suite="bbob_noisy",
        maker=Partial(
            _make_gallagher,
            fn=bbob_noisy.f129,
            num_peaks=101,
            w_divisor=99,
            alpha_first=1000.0,
            y_minval=-5.0,
            y_maxval=5.0,
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="uniform", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.gallagher_true,
    ),
    FunctionSpec(
        name="bbob_noisy_f130",
        suite="bbob_noisy",
        maker=Partial(
            _make_gallagher,
            fn=bbob_noisy.f130,
            num_peaks=101,
            w_divisor=99,
            alpha_first=1000.0,
            y_minval=-5.0,
            y_maxval=5.0,
        ),
        tags=_bbob_noisy_tags(
            separable=False, unimodal=False, model="cauchy", severe=True
        ),
        bounds=BBOB_BOUNDS,
        true_fn=bbob_noisy.gallagher_true,
    ),
)

#                                                               CEC 2005 table
# =============================================================================

_NC = 10  # num_components for all composition functions (F15-F25)

# Lambda values shared by composition function groups. Stored as plain
# Python tuples (not jnp.array) so that import order does not bake in a
# dtype before the user has had a chance to call
# ``jax.config.update("jax_enable_x64", True)``. The factories convert
# them to jax.Array at call time via ``jnp.asarray(..., dtype=float)``.
_COMP1_LAMBDA: tuple[float, ...] = (
    1,
    1,
    10,
    10,
    5 / 60,
    5 / 60,
    5 / 32,
    5 / 32,
    5 / 100,
    5 / 100,
)
_COMP2_LAMBDA_F18: tuple[float, ...] = (
    2 * 5 / 32,
    5 / 32,
    2 * 1,
    1,
    2 * 5 / 100,
    5 / 100,
    2 * 10,
    10,
    2 * 5 / 60,
    5 / 60,
)
_COMP2_LAMBDA_F19: tuple[float, ...] = (
    0.1 * 5 / 32,
    5 / 32,
    2 * 1,
    1,
    2 * 5 / 100,
    5 / 100,
    2 * 10,
    10,
    2 * 5 / 60,
    5 / 60,
)
_COMP3_LAMBDA: tuple[float, ...] = (
    5 * 5 / 100,
    5 / 100,
    5 * 1,
    1,
    5 * 1,
    1,
    5 * 10,
    10,
    5 * 5 / 200,
    5 / 200,
)

CEC2005_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        name="f1",
        suite="cec2005",
        maker=Partial(
            make_cec2005,
            fn=f1,
            num_components=1,
            minval=-100.0,
            maxval=100.0,
        ),
        tags=_cec_tags(unimodal=True),
        bounds=(-100.0, 100.0),
    ),
    FunctionSpec(
        name="f2",
        suite="cec2005",
        maker=Partial(
            make_cec2005,
            fn=f2,
            num_components=1,
            minval=-100.0,
            maxval=100.0,
        ),
        tags=_cec_tags(unimodal=True),
        bounds=(-100.0, 100.0),
    ),
    FunctionSpec(
        name="f3",
        suite="cec2005",
        maker=Partial(
            make_cec2005,
            fn=f3,
            num_components=1,
            minval=-100.0,
            maxval=100.0,
        ),
        tags=_cec_tags(unimodal=True, rotated=True),
        bounds=(-100.0, 100.0),
    ),
    FunctionSpec(
        name="f4",
        suite="cec2005",
        maker=Partial(
            make_cec2005,
            fn=f4,
            num_components=1,
            minval=-100.0,
            maxval=100.0,
        ),
        tags=_cec_tags(unimodal=True, noise=True),
        bounds=(-100.0, 100.0),
        true_fn=f4_true,
    ),
    FunctionSpec(
        name="f5",
        suite="cec2005",
        maker=Partial(_make_cec2005_f5, fn=f5, num_components=1),
        tags=_cec_tags(unimodal=True),
        bounds=(-100.0, 100.0),
    ),
    FunctionSpec(
        name="f6",
        suite="cec2005",
        maker=Partial(
            make_cec2005,
            fn=f6,
            num_components=1,
            minval=-100.0,
            maxval=100.0,
        ),
        tags=_cec_tags(),
        bounds=(-100.0, 100.0),
    ),
    FunctionSpec(
        name="f7",
        suite="cec2005",
        maker=Partial(_make_cec2005_f7, fn=f7, num_components=1),
        tags=_cec_tags(rotated=True),
        bounds=(0.0, 600.0),
    ),
    FunctionSpec(
        name="f8",
        suite="cec2005",
        maker=Partial(_make_cec2005_f8, fn=f8, num_components=1),
        tags=_cec_tags(rotated=True),
        bounds=(-32.0, 32.0),
    ),
    FunctionSpec(
        name="f9",
        suite="cec2005",
        maker=Partial(
            make_cec2005, fn=f9, num_components=1, minval=-5.0, maxval=5.0
        ),
        tags=_cec_tags(),
        bounds=(-5.0, 5.0),
    ),
    FunctionSpec(
        name="f10",
        suite="cec2005",
        maker=Partial(
            _make_cec2005_conditioned_single,
            fn=f10,
            minval=-5.0,
            maxval=5.0,
            condition_number=2.0,
        ),
        tags=_cec_tags(rotated=True),
        bounds=(-5.0, 5.0),
    ),
    FunctionSpec(
        name="f11",
        suite="cec2005",
        maker=Partial(
            _make_cec2005_conditioned_single,
            fn=f11,
            minval=-0.5,
            maxval=0.5,
            condition_number=5.0,
        ),
        tags=_cec_tags(rotated=True),
        bounds=(-0.5, 0.5),
    ),
    FunctionSpec(
        name="f12",
        suite="cec2005",
        maker=Partial(_make_cec2005_f12, fn=f12, num_components=1),
        tags=_cec_tags(),
        bounds=(-math.pi, math.pi),
    ),
    FunctionSpec(
        name="f13",
        suite="cec2005",
        maker=Partial(
            make_cec2005, fn=f13, num_components=1, minval=-3.0, maxval=1.0
        ),
        tags=_cec_tags(),
        bounds=(-3.0, 1.0),
    ),
    FunctionSpec(
        name="f14",
        suite="cec2005",
        maker=Partial(
            _make_cec2005_conditioned_single,
            fn=f14,
            minval=-100.0,
            maxval=100.0,
            condition_number=3.0,
        ),
        tags=_cec_tags(rotated=True),
        bounds=(-100.0, 100.0),
    ),
    FunctionSpec(
        name="f15",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=make_cec2005,
            comp_fns_builder=_composition1_fns,
            comp_lambda=_COMP1_LAMBDA,
            fn=f15,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
        ),
        tags=_cec_tags(composition=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f16",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=make_cec2005_conditioned,
            comp_fns_builder=_composition1_fns,
            comp_lambda=_COMP1_LAMBDA,
            fn=f16,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
            condition_numbers=(2.0,) * _NC,
        ),
        tags=_cec_tags(composition=True, rotated=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f17",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=make_cec2005,
            comp_fns_builder=_composition1_fns,
            comp_lambda=_COMP1_LAMBDA,
            fn=f17,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
        ),
        tags=_cec_tags(composition=True, rotated=True, noise=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
        true_fn=f16,
    ),
    FunctionSpec(
        name="f18",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=_make_cec2005_f18_family,
            comp_fns_builder=_composition2_fns,
            comp_lambda=_COMP2_LAMBDA_F18,
            fn=f18,
            num_components=_NC,
        ),
        tags=_cec_tags(composition=True, rotated=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f19",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=_make_cec2005_f18_family,
            comp_fns_builder=_composition2_fns,
            comp_lambda=_COMP2_LAMBDA_F19,
            fn=f19,
            num_components=_NC,
        ),
        tags=_cec_tags(composition=True, rotated=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f20",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=_make_cec2005_f20,
            comp_fns_builder=_composition2_fns,
            comp_lambda=_COMP2_LAMBDA_F18,
            fn=f20,
            num_components=_NC,
        ),
        tags=_cec_tags(composition=True, rotated=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f21",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=make_cec2005_conditioned,
            comp_fns_builder=_composition3_fns,
            comp_lambda=_COMP3_LAMBDA,
            fn=f21,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
            condition_numbers=(1.0,) * _NC,
        ),
        tags=_cec_tags(composition=True, rotated=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f22",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=make_cec2005_conditioned,
            comp_fns_builder=_composition3_fns,
            comp_lambda=_COMP3_LAMBDA,
            fn=f22,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
            condition_numbers=(
                10.0,
                20.0,
                50.0,
                100.0,
                200.0,
                1000.0,
                2000.0,
                3000.0,
                4000.0,
                5000.0,
            ),
        ),
        tags=_cec_tags(composition=True, rotated=True),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f23",
        suite="cec2005",
        maker=Partial(
            _add_f_max,
            base_factory=make_cec2005_conditioned,
            comp_fns_builder=_composition3_fns,
            comp_lambda=_COMP3_LAMBDA,
            fn=f23,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
            condition_numbers=(1.0,) * _NC,
        ),
        tags=_cec_tags(
            composition=True, rotated=True, structure_modified=True
        ),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="f24",
        suite="cec2005",
        maker=Partial(
            make_cec2005_conditioned,
            fn=f24,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
            condition_numbers=(
                100.0,
                50.0,
                30.0,
                10.0,
                5.0,
                5.0,
                4.0,
                3.0,
                2.0,
                2.0,
            ),
        ),
        tags=_cec_tags(
            composition=True,
            rotated=True,
            noise=True,
            structure_modified=True,
        ),
        bounds=(-5.0, 5.0),
        x_opt_from=_first_component_x_opt,
        true_fn=f24_true,
    ),
    FunctionSpec(
        name="f25",
        suite="cec2005",
        maker=Partial(
            make_cec2005_conditioned,
            fn=f25,
            num_components=_NC,
            minval=-5.0,
            maxval=5.0,
            condition_numbers=(
                100.0,
                50.0,
                30.0,
                10.0,
                5.0,
                5.0,
                4.0,
                3.0,
                2.0,
                2.0,
            ),
        ),
        tags=_cec_tags(
            composition=True,
            rotated=True,
            noise=True,
            structure_modified=True,
        ),
        bounds=(2.0, 5.0),
        x_opt_from=_first_component_x_opt,
        true_fn=f24_true,
    ),
)


#                                                               CEC 2017 table
# =============================================================================
# F1-F10 (simple functions); the hybrids F11-F20 and compositions F21-F30
# follow in later waves. F2 was officially withdrawn and is not implemented;
# the numbering keeps the hole, matching the reference code and data files.

CEC2017_BOUNDS: tuple[float, float] = (-100.0, 100.0)

CEC2017_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        name="cec2017_f1",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f1),
        tags=_cec2017_tags(unimodal=True),
        bounds=CEC2017_BOUNDS,
    ),
    FunctionSpec(
        name="cec2017_f3",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f3),
        tags=_cec2017_tags(unimodal=True),
        bounds=CEC2017_BOUNDS,
    ),
    FunctionSpec(
        name="cec2017_f4",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f4),
        tags=_cec2017_tags(),
        bounds=CEC2017_BOUNDS,
    ),
    FunctionSpec(
        name="cec2017_f5",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f5),
        tags=_cec2017_tags(),
        bounds=CEC2017_BOUNDS,
    ),
    FunctionSpec(
        name="cec2017_f6",
        suite="cec2017",
        # Rotation is dead code in the official reference (see cec2017.f6),
        # so the instance is shift-only: rotated=False.
        maker=Partial(make_cec2017, fn=cec2017.f6, min_ndim=2),
        tags=_cec2017_tags(rotated=False),
        bounds=CEC2017_BOUNDS,
        min_ndim=2,
    ),
    FunctionSpec(
        name="cec2017_f7",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f7),
        tags=_cec2017_tags(),
        bounds=CEC2017_BOUNDS,
    ),
    FunctionSpec(
        name="cec2017_f8",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f8),
        tags=_cec2017_tags(),
        bounds=CEC2017_BOUNDS,
    ),
    FunctionSpec(
        name="cec2017_f9",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f9),
        tags=_cec2017_tags(),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_cec2017_levy_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f10",
        suite="cec2017",
        maker=Partial(make_cec2017, fn=cec2017.f10),
        tags=_cec2017_tags(),
        bounds=CEC2017_BOUNDS,
    ),
    # Hybrids F11-F20: min_ndim is at least one dimension per subcomponent
    # kernel; f14 and f20 need more because their Schaffer F7 chunk must
    # hold two dimensions under the chunk-partition rule (see
    # cec2017_hybrid_partition).
    FunctionSpec(
        name="cec2017_f11",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f11, min_ndim=3),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=3,
    ),
    FunctionSpec(
        name="cec2017_f12",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f12, min_ndim=3),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=3,
    ),
    FunctionSpec(
        name="cec2017_f13",
        suite="cec2017",
        # min_ndim 4, not 3: the Lunacek chunk needs two dimensions
        # (its depth factor is negative at one — NaN in the reference).
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f13, min_ndim=4),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=4,
    ),
    FunctionSpec(
        name="cec2017_f14",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f14, min_ndim=6),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=6,
    ),
    FunctionSpec(
        name="cec2017_f15",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f15, min_ndim=4),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=4,
    ),
    FunctionSpec(
        name="cec2017_f16",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f16, min_ndim=4),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=4,
    ),
    FunctionSpec(
        name="cec2017_f17",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f17, min_ndim=5),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=5,
    ),
    FunctionSpec(
        name="cec2017_f18",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f18, min_ndim=5),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=5,
    ),
    FunctionSpec(
        name="cec2017_f19",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f19, min_ndim=5),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=5,
    ),
    FunctionSpec(
        name="cec2017_f20",
        suite="cec2017",
        maker=Partial(make_cec2017_hybrid, fn=cec2017.f20, min_ndim=7),
        tags=_cec2017_tags(hybrid=True),
        bounds=CEC2017_BOUNDS,
        min_ndim=7,
    ),
    # Compositions F21-F30: the global minimum is the first component's
    # shift; deterministic instances are degenerate (same caveat as the
    # deterministic CEC 2005 compositions). F29/F30 compose full hybrids
    # and inherit the largest component hybrid's min_ndim.
    FunctionSpec(
        name="cec2017_f21",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f21, num_components=3
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f22",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f22, num_components=3
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f23",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f23, num_components=4
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f24",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f24, num_components=4
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f25",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f25, num_components=5
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f26",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f26, num_components=5
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f27",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f27, num_components=6
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f28",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition, fn=cec2017.f28, num_components=6
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
    ),
    FunctionSpec(
        name="cec2017_f29",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition,
            fn=cec2017.f29,
            num_components=3,
            with_shuffles=True,
            min_ndim=5,
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
        min_ndim=5,
    ),
    FunctionSpec(
        name="cec2017_f30",
        suite="cec2017",
        maker=Partial(
            make_cec2017_composition,
            fn=cec2017.f30,
            num_components=3,
            with_shuffles=True,
            min_ndim=5,
        ),
        tags=_cec2017_tags(composition=True),
        bounds=CEC2017_BOUNDS,
        x_opt_from=_first_component_x_opt,
        min_ndim=5,
    ),
)


#                                                                Derived views
# =============================================================================

SPECS: tuple[FunctionSpec, ...] = (
    BBOB_SPECS + BBOB_NOISY_SPECS + CEC2005_SPECS + CEC2017_SPECS
)
SPEC_BY_NAME: dict[str, FunctionSpec] = {s.name: s for s in SPECS}

if len(SPEC_BY_NAME) != len(SPECS):
    raise RuntimeError("duplicate function names in SPECS")
