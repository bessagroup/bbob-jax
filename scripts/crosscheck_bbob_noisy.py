"""Cross-validate the BBOB-noisy suite against the official reference code.

One-off development validation (not part of CI — instances in bbob-jax are
seed-generated, so this script instead *injects reference-derived instance
parameters* into the bbob-jax function definitions and compares the
deterministic undisturbed path (Ftrue) point-for-point against the compiled
legacy C code):

1. Compiles the legacy ``benchmarksnoisy.c`` (+ helpers) with a tiny driver
   (needs ``gcc``). The driver prints ``Fopt``, ``Xopt`` and the undisturbed
   ``Ftrue`` at probe points for a given (function, dimension, instance).
2. Extracts the same instance's parameters (shift, rotations, Gallagher
   peaks) from the official legacy Python ``bbobbenchmarks.py`` — the code
   COCO's own noisy regression test uses — converting from its row-vector
   convention (transposing rotations) to the column-vector convention of
   the C code and bbob-jax.
3. Binds those parameters into the bbob-jax ``*_true`` implementations via
   ``jax.tree_util.Partial`` — exactly the parameter slots the factories
   normally fill with sampled values — and evaluates on the same probe
   points in ``[-5.5, 5.5]^D`` (float64; deliberately beyond the bounds to
   exercise the x100 boundary penalty).
4. Reports the max absolute/relative deviation per function, for both
   legacy-Python-vs-C (extraction sanity) and bbob-jax-vs-C.

The noisy path itself cannot be compared bitwise (the C reference uses its
own RNG); it is covered by the pinned-draw and statistical tests in
``tests/test_bbob_noisy.py``.

Usage::

    uv run python scripts/crosscheck_bbob_noisy.py \\
        --ref-dir /path/to/bbob-legacy-code/c \\
        --legacy-py /path/to/bbobbenchmarks.py
    # ref-dir: the ``c/`` directory of the legacy BBOB code (contains
    #   benchmarksnoisy.c, benchmarkshelper.c, benchmarksdeclare.c, ...);
    #   mirrored e.g. at github.com/lorenzo-consoli/bbob-legacy-code.
    # legacy-py: the official legacy Python implementation, in numbbo/coco
    #   at code-postprocessing/aRTAplots/bbobbenchmarks.py.

Options: ``--dims 2 5 10`` ``--instances 1 4`` ``--n-points 12``
``--seed 0``.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

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
    double *ft = (double *)malloc(sizeof(double) * npoints);
    for (int p = 0; p < npoints; p++) {
        for (int i = 0; i < dim; i++)
            if (scanf("%lf", &x[i]) != 1)
                return 2;
        TwoDoubles r = fn(x);
        ft[p] = r.Ftrue;
    }
    printf("FOPT %.17g\n", Fopt);
    printf("XOPT");
    for (int i = 0; i < dim; i++)
        printf(" %.17g", Xopt[i]);
    printf("\n");
    for (int p = 0; p < npoints; p++)
        printf("%.17g\n", ft[p]);
    finibenchmarksnoisy();
    finibenchmarkshelper();
    return 0;
}
"""

REF_SOURCES = [
    "benchmarksnoisy.c",
    "benchmarkshelper.c",
    "benchmarksdeclare.c",
]

FUNC_NUMS = list(range(101, 131))

# fid -> (bbob-jax *_true implementation name, parameter-binding style)
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


def compile_reference(ref_dir: Path, build_dir: Path) -> Path:
    for name in REF_SOURCES:
        if not (ref_dir / name).exists():
            sys.exit(f"missing {name} in {ref_dir}")
    driver = build_dir / "driver.c"
    driver.write_text(DRIVER_C)
    exe = build_dir / "bbob_noisy_ref"
    sources = [str(driver)] + [str(ref_dir / s) for s in REF_SOURCES]
    subprocess.run(
        ["gcc", "-O2", "-I", str(ref_dir), "-o", str(exe)] + sources + ["-lm"],
        check=True,
    )
    return exe


def run_reference(
    exe: Path, fid: int, dim: int, instance: int, points: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (fopt, xopt, ftrue-at-points) from the compiled C code."""
    payload = "\n".join(
        " ".join(format(v, ".17g") for v in row) for row in points
    )
    out = subprocess.run(
        [str(exe), str(fid), str(dim), str(instance), str(len(points))],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    fopt = float(out[0].split()[1])
    xopt = np.array([float(v) for v in out[1].split()[1:]])
    ftrue = np.array([float(v) for v in out[2:]])
    return fopt, xopt, ftrue


def load_legacy_python(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bbobbenchmarks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_jax_params(style: str, legacy: Any, dim: int) -> dict:
    """Map legacy-Python instance attributes to bbob-jax keyword slots.

    The legacy Python evaluates row vectors (``x @ M``); bbob-jax and the
    C code evaluate column vectors (``M @ x``), so every extracted matrix
    is transposed.
    """
    import jax.numpy as jnp

    eye = jnp.eye(dim, dtype=jnp.float64)
    as_j = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)  # noqa
    kw = {
        "x_opt": as_j(legacy.xopt),
        "f_opt": as_j(legacy.fopt),
        "R": eye,
        "Q": eye,
    }
    if style == "shift_only":
        pass
    elif style == "rotated":
        kw["R"] = as_j(legacy.rotation).T
    elif style == "rotated_noshift":
        # griewank-rosenbrock: the legacy linearTF folds the scale into
        # the rotation; bbob-jax applies the scale itself.
        scale = max(1, dim**0.5 / 8.0)
        kw["x_opt"] = jnp.zeros(dim, dtype=jnp.float64)
        kw["R"] = as_j(legacy.linearTF).T / scale
    elif style == "mat_rot":
        # step ellipsoid: linearTF (applied first) and rotation (after
        # rounding) — bbob-jax slots _mat and Q.
        kw["Q"] = as_j(legacy.rotation).T
        kw["_mat"] = as_j(legacy.linearTF).T
    elif style == "rot_mat":
        # schaffer F7: rotation (applied first) and linearTF (after the
        # asymmetric transform) — bbob-jax slots R and _mat.
        kw["R"] = as_j(legacy.rotation).T
        kw["_mat"] = as_j(legacy.linearTF).T
    elif style == "gallagher":
        kw["R"] = as_j(legacy.rotation).T
        kw["_gal_w"] = as_j(legacy.peakvalues)
        kw["_gal_y_rot"] = as_j(legacy.xlocal)
        kw["_gal_c_diags"] = as_j(legacy.arrscales)
    else:
        raise ValueError(style)
    return kw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--legacy-py", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--instances", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--n-points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel-tol", type=float, default=1e-11)
    args = parser.parse_args()

    import tempfile

    import jax

    jax.config.update("jax_enable_x64", True)
    from jax.tree_util import Partial

    from bbob_jax._src import bbob_noisy

    bn = load_legacy_python(args.legacy_py)

    worst_py = 0.0
    worst_jax = 0.0
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        exe = compile_reference(args.ref_dir, Path(tmp))
        rng = np.random.default_rng(args.seed)
        for dim in args.dims:
            for instance in args.instances:
                points = rng.uniform(-5.5, 5.5, size=(args.n_points, dim))
                for fid in FUNC_NUMS:
                    fopt_c, xopt_c, ftrue_c = run_reference(
                        exe, fid, dim, instance, points
                    )
                    legacy = getattr(bn, f"F{fid}")(instance)
                    _, ftrue_py = legacy._evalfull(points.copy())
                    scale = np.maximum(np.abs(ftrue_c), 1.0)
                    dev_py = np.max(np.abs(ftrue_py - ftrue_c) / scale)
                    worst_py = max(worst_py, dev_py)

                    impl_name, style = BASES[fid]
                    kw = bind_jax_params(style, legacy, dim)
                    fn = Partial(getattr(bbob_noisy, impl_name), **kw)
                    ftrue_jax = np.array([float(fn(p)) for p in points])
                    dev_jax = np.max(np.abs(ftrue_jax - ftrue_c) / scale)
                    worst_jax = max(worst_jax, dev_jax)

                    status = "ok"
                    if dev_jax > args.rel_tol or dev_py > args.rel_tol:
                        status = "FAIL"
                        failures.append((fid, dim, instance))
                    print(
                        f"f{fid} D={dim} i={instance}: "
                        f"py-vs-C {dev_py:.2e}  jax-vs-C {dev_jax:.2e}  "
                        f"{status}"
                    )
    print(f"\nworst py-vs-C  rel dev: {worst_py:.3e}")
    print(f"worst jax-vs-C rel dev: {worst_jax:.3e}")
    if failures:
        sys.exit(f"FAILURES: {failures}")
    print("all comparisons within tolerance")


if __name__ == "__main__":
    main()
