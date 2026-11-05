"""Cross-check of the BBOB-noisy suite against the legacy reference code.

bbob-jax draws instance parameters from JAX keys, so the test injects
reference parameters and compares the undisturbed value (``Ftrue``)
point for point with the compiled legacy C ``benchmarksnoisy.c``:

1. A small driver around the legacy C prints ``Ftrue`` at given points
   for a (function, dimension, instance).
2. The same instance's parameters (shift, rotations, Gallagher peaks)
   come from COCO's legacy Python ``bbobbenchmarks.py``, the code COCO's
   own noisy regression test uses. Its rotations are transposed to the
   column-vector convention of the C code and bbob-jax.
3. They are bound into the bbob-jax ``*_true`` implementations through
   the keyword slots the factories normally fill, and both are evaluated
   in float64 on the same points in [-5.5, 5.5]^D, past the bounds on
   purpose to exercise the x100 boundary penalty.

Both the legacy Python and bbob-jax must match the C within 1e-11
relative. Coverage: all 30 functions, D in {2, 5, 10}, instances 1 and
4, 12 points each. Worst relative deviation is a few 1e-12
(Griewank-Rosenbrock, summation order).

The noisy path itself cannot be compared bitwise, because the C uses its
own random number generator; ``test_bbob_noisy.py`` covers it with
pinned-draw and statistical tests. Needs the reference (see
``tests/_references.py``) and skips without it.
"""

import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial

from bbob_jax._src import bbob_noisy
from tests._references import (
    LEGACY_C,
    LEGACY_PY,
    build,
    compiler,
    load_legacy_bbob,
    reference_path,
    relative_deviation,
)

DIMS = (2, 5, 10)
INSTANCES = (1, 4)
N_POINTS = 12
REL_TOL = 1e-11

DRIVER_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "bbobStructures.h"
#include "benchmarksnoisy.h"

extern int DIM;
extern int trialid;
extern unsigned int isInitDone;
extern double Fopt;
extern double *Xopt;
extern void initbenchmarkshelper(void);
extern void finibenchmarkshelper(void);
extern void initbenchmarksnoisy(void);
extern void finibenchmarksnoisy(void);

int main(int argc, char **argv)
{
    if (argc != 5)
        return 1;
    int fid = atoi(argv[1]);
    int dim = atoi(argv[2]);
    int instance = atoi(argv[3]);
    int npoints = atoi(argv[4]);
    DIM = dim;
    trialid = instance;
    initbenchmarkshelper();
    initbenchmarksnoisy();
    isInitDone = 0;
    bbobFunction fn = handlesNoisy[fid - 101];
    double *x = (double *)malloc(sizeof(double) * dim);
    for (int p = 0; p < npoints; p++) {
        for (int i = 0; i < dim; i++)
            if (scanf("%lf", &x[i]) != 1)
                return 2;
        TwoDoubles r = fn(x);
        printf("%.17g\n", r.Ftrue);
    }
    finibenchmarksnoisy();
    finibenchmarkshelper();
    return 0;
}
"""

REF_SOURCES = (
    "benchmarksnoisy.c",
    "benchmarkshelper.c",
    "benchmarksdeclare.c",
)

# fid -> (bbob-jax *_true implementation, parameter-binding style)
BASES: dict[int, tuple[str, str]] = {
    **{fid: ("sphere_true", "shift_only") for fid in (101, 102, 103)},
    **{fid: ("rosenbrock_true", "shift_only") for fid in (104, 105, 106)},
    **{fid: ("sphere_true", "shift_only") for fid in (107, 108, 109)},
    **{fid: ("rosenbrock_true", "shift_only") for fid in (110, 111, 112)},
    **{fid: ("step_ellipsoid_true", "mat_rot") for fid in (113, 114, 115)},
    **{fid: ("ellipsoid_true", "rotated") for fid in (116, 117, 118)},
    **{fid: ("different_powers_true", "rotated") for fid in (119, 120, 121)},
    **{fid: ("schaffer_f7_true", "rot_mat") for fid in (122, 123, 124)},
    **{
        fid: ("griewank_rosenbrock_true", "rotated_noshift")
        for fid in (125, 126, 127)
    },
    **{fid: ("gallagher_true", "gallagher") for fid in (128, 129, 130)},
}


@pytest.fixture(scope="module")
def legacy():
    return load_legacy_bbob(reference_path(LEGACY_PY))


@pytest.fixture(scope="module")
def reference_exe(tmp_path_factory) -> Path:
    ref_dir = reference_path(LEGACY_C)
    gcc = compiler("gcc")
    build_dir = tmp_path_factory.mktemp("bbob_noisy_ref")
    driver = build_dir / "driver.c"
    driver.write_text(DRIVER_C)
    exe = build_dir / "bbob_noisy_ref"
    sources = [str(driver)] + [str(ref_dir / s) for s in REF_SOURCES]
    build([gcc, "-O2", "-I", str(ref_dir), "-o", str(exe), *sources, "-lm"])
    return exe


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _ftrue_from_c(
    exe: Path, fid: int, dim: int, instance: int, points: np.ndarray
) -> np.ndarray:
    payload = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in points)
    out = subprocess.run(
        [str(exe), str(fid), str(dim), str(instance), str(len(points))],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return np.array([float(v) for v in out.stdout.split()])


def _bind(style: str, ref: Any, dim: int) -> dict:
    """Map the legacy Python's instance attributes to bbob-jax slots."""

    def f64(a: Any) -> jax.Array:
        return jnp.asarray(np.asarray(a), dtype=jnp.float64)

    eye = jnp.eye(dim, dtype=jnp.float64)
    kw = {"x_opt": f64(ref.xopt), "f_opt": f64(ref.fopt), "R": eye, "Q": eye}
    if style == "rotated":
        kw["R"] = f64(ref.rotation).T
    elif style == "rotated_noshift":
        # Griewank-Rosenbrock: the reference folds the scale into
        # linearTF; bbob-jax applies the scale itself.
        kw["x_opt"] = jnp.zeros(dim, dtype=jnp.float64)
        kw["R"] = f64(ref.linearTF).T / max(1, dim**0.5 / 8.0)
    elif style == "mat_rot":
        # step ellipsoid: linearTF first, rotation after rounding.
        kw["Q"] = f64(ref.rotation).T
        kw["_mat"] = f64(ref.linearTF).T
    elif style == "rot_mat":
        # Schaffer F7: rotation first, linearTF after the asymmetry.
        kw["R"] = f64(ref.rotation).T
        kw["_mat"] = f64(ref.linearTF).T
    elif style == "gallagher":
        kw["R"] = f64(ref.rotation).T
        kw["_gal_w"] = f64(ref.peakvalues)
        kw["_gal_y_rot"] = f64(ref.xlocal)
        kw["_gal_c_diags"] = f64(ref.arrscales)
    elif style != "shift_only":
        raise ValueError(style)
    return kw


@pytest.mark.parametrize("fid", sorted(BASES), ids=lambda fid: f"f{fid}")
def test_undisturbed_value_matches_legacy_c(fid, legacy, reference_exe):
    impl, style = BASES[fid]
    worst_py = worst_jax = 0.0
    for dim in DIMS:
        for instance in INSTANCES:
            rng = np.random.default_rng([dim, instance])
            points = rng.uniform(-5.5, 5.5, size=(N_POINTS, dim))
            f_c = _ftrue_from_c(reference_exe, fid, dim, instance, points)
            ref = getattr(legacy, f"F{fid}")(instance)
            _, f_py = ref._evalfull(points.copy())
            fn = Partial(getattr(bbob_noisy, impl), **_bind(style, ref, dim))
            f_jax = np.array([float(fn(jnp.asarray(p))) for p in points])
            worst_py = max(worst_py, relative_deviation(f_py, f_c))
            worst_jax = max(worst_jax, relative_deviation(f_jax, f_c))
    print(f"f{fid}: legacy Python {worst_py:.2e}, bbob-jax {worst_jax:.2e}")
    assert worst_py <= REL_TOL, f"parameter extraction off: {worst_py:.2e}"
    assert worst_jax <= REL_TOL, f"f{fid} {impl}: {worst_jax:.2e}"
