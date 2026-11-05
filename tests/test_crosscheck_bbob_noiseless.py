"""Cross-check of the noiseless BBOB suite against COCO's reference code.

bbob-jax draws instance parameters from JAX keys, not COCO instance IDs.
So the test takes each instance's parameters (shift, rotations, linear
transforms, Gallagher peaks) from COCO's legacy Python
``bbobbenchmarks.py``, binds them into the bbob-jax implementations
through the keyword slots the factories normally fill, and compares the
two point for point in float64.

Coverage: all 24 functions, COCO instances 1-5, D in {5, 10}, 100
random points in [-5, 5]^D each (the same points for every instance;
the instance moves the optimum and rotations under them). The legacy
Python works with row vectors, so matrices are transposed where
bbob-jax uses column vectors. Worst relative deviation is about 1.3e-10
(F19 on instances 1 and 4, summation order); every other function stays
below 2e-11.

The legacy Python is itself the code COCO's regression tests compare
with the legacy C; ``test_crosscheck_bbob_noisy.py`` also pins it to the
compiled C on the shared machinery. This check caught the deviations
fixed in ADR 0005. It needs the reference (see ``tests/_references.py``)
and skips without it.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial

import bbob_jax._src.bbob as bbob
from bbob_jax._src.transforms import lambda_func
from tests._references import (
    LEGACY_PY,
    load_legacy_bbob,
    reference_path,
    relative_deviation,
)

INSTANCES = (1, 2, 3, 4, 5)
DIMS = (5, 10)
N_POINTS = 100
REL_TOL = 1e-9


@pytest.fixture(scope="module")
def legacy():
    return load_legacy_bbob(reference_path(LEGACY_PY))


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _f64(a: Any) -> jax.Array:
    return jnp.asarray(np.asarray(a), dtype=jnp.float64)


def _linear_slope(ref: Any) -> dict:
    x_opt = np.asarray(ref.xopt)
    s = np.sign(x_opt) * 10.0 ** np.linspace(0, 1, len(x_opt))
    return {"_ls_x_opt": _f64(x_opt), "_ls_s": _f64(s)}


def _scaled_rotation(ref: Any, transpose: bool) -> dict:
    # F9/F19: the reference folds the scale into linearTF; bbob-jax
    # applies the scale itself and adds an (optional) shift.
    dim = np.asarray(ref.linearTF).shape[0]
    scale = max(1, dim**0.5 / 8.0)
    mat = _f64(ref.linearTF)
    return {
        "x_opt": jnp.zeros(dim, dtype=jnp.float64),
        "R": (mat.T if transpose else mat) / scale,
    }


def _schwefel(ref: Any) -> dict:
    dim = len(np.asarray(ref.xopt))
    return {
        "_sw_ones": _f64(np.sign(np.asarray(ref.xopt))),
        "_sw_x_opt_shape": _f64(ref.xopt),
        "_sw_lamb": lambda_func(dim, 10.0),
    }


def _gallagher(ref: Any) -> dict:
    return {
        "R": _f64(ref.rotation).T,
        "_gal_w": _f64(ref.peakvalues),
        "_gal_y_rot": _f64(ref.xlocal),
        "_gal_c_diags": _f64(ref.arrscales),
    }


def _mat(ref: Any) -> dict:
    return {"_mat": _f64(ref.linearTF).T}


def _rot(ref: Any) -> dict:
    return {"R": _f64(ref.rotation).T}


def _rot_mat(ref: Any) -> dict:
    return {"R": _f64(ref.rotation).T, "_mat": _f64(ref.linearTF).T}


def _nothing(ref: Any) -> dict:
    return {}


# COCO function id -> (bbob-jax implementation, parameter binder)
SETUPS: dict[int, tuple[str, Callable[[Any], dict]]] = {
    1: ("sphere", _nothing),
    2: ("ellipsoid_seperable", _nothing),
    3: ("rastrigin_seperable", _nothing),
    4: ("skew_rastrigin_bueche", _nothing),
    5: ("linear_slope", _linear_slope),
    6: ("attractive_sector", _mat),
    7: (
        "step_ellipsoid",
        lambda ref: {"_mat": _f64(ref.linearTF).T, "Q": _f64(ref.rotation).T},
    ),
    8: ("rosenbrock", _nothing),
    9: ("rosenbrock_rotated", lambda ref: _scaled_rotation(ref, False)),
    # bbob-jax uses the row convention (x - x_opt) @ R here
    10: ("ellipsoid", lambda ref: {"R": _f64(ref.rotation)}),
    11: ("discuss", _rot),
    12: ("bent_cigar", _rot),
    13: ("sharp_ridge", _mat),
    14: ("sum_of_different_powers", _rot),
    15: ("rastrigin", _rot_mat),
    16: ("weierstrass", _rot_mat),
    17: ("schaffer_f7_condition_10", _rot_mat),
    18: ("schaffer_f7_condition_1000", _rot_mat),
    19: (
        "griewank_rosenbrock_f8f2",
        lambda ref: _scaled_rotation(ref, True),
    ),
    20: ("schwefel_xsinx", _schwefel),
    21: ("gallagher_101_peaks", _gallagher),
    22: ("gallagher_21_peaks", _gallagher),
    23: ("katsuura", _mat),
    24: (
        "lunacek_bi_rastrigin",
        lambda ref: {
            "_mat": _f64(ref.linearTF).T,
            "_x_opt_shape": _f64(ref.xopt),
        },
    ),
}


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda i: f"i{i}")
@pytest.mark.parametrize(
    "fid", sorted(SETUPS), ids=lambda fid: f"f{fid}-{SETUPS[fid][0]}"
)
def test_matches_coco_reference(fid, instance, legacy):
    name, binder = SETUPS[fid]
    worst = worst_abs = 0.0
    for dim in DIMS:
        rng = np.random.default_rng(fid * 100 + dim)
        points = rng.uniform(-5, 5, size=(N_POINTS, dim))
        ref = getattr(legacy, f"F{fid}")(instance)
        _, f_ref = ref._evalfull(points.copy())
        eye = jnp.eye(dim, dtype=jnp.float64)
        kw = {"x_opt": _f64(ref.xopt), "f_opt": _f64(ref.fopt)}
        kw.update({"R": eye, "Q": eye})
        kw.update(binder(ref))
        fn = Partial(getattr(bbob, name), **kw)
        f_jax = np.array([float(fn(_f64(p))) for p in points])
        worst = max(worst, relative_deviation(f_jax, f_ref))
        worst_abs = max(worst_abs, float(np.max(np.abs(f_jax - f_ref))))
    print(
        f"F{fid} instance {instance} {name}: max relative deviation "
        f"{worst:.2e}, max absolute deviation {worst_abs:.2e}"
    )
    assert worst <= REL_TOL, (
        f"F{fid} instance {instance} {name}: {worst:.2e} > {REL_TOL:.0e}"
    )
