"""Cross-validate the noiseless BBOB suite against the official reference.

One-off development validation (not part of CI — instances in bbob-jax are
seed-generated, so this script instead *injects reference-derived instance
parameters* into the bbob-jax function definitions):

1. Extracts each instance's parameters (shift, rotations, linear
   transforms, Gallagher peaks) from the official legacy Python
   ``bbobbenchmarks.py`` — converting from its row-vector convention
   (transposing matrices) to the column-vector convention of bbob-jax
   where needed.
2. Binds them into the bbob-jax implementations via
   ``jax.tree_util.Partial`` — exactly the parameter slots the factories
   normally fill with sampled values — and compares point-for-point on
   random points in ``[-5, 5]^D`` (float64) against the legacy values.

The legacy Python is itself the code COCO's regression tests compare
against the legacy C; the noisy-suite harness
(``scripts/crosscheck_bbob_noisy.py``) additionally pins it to the
compiled C on the shared machinery.

This audit is what caught the pre-ADR-0005 deviations (T_osz mask and
constants, T_asy off-by-one, F3/F15 product core, F4 skew parity, F14
missing sqrt); after the fix every function matches to ~1e-13 (F19
~1e-10, accumulation order).

Usage::

    uv run python scripts/crosscheck_bbob_noiseless.py \\
        --legacy-py /path/to/bbobbenchmarks.py
    # legacy-py: the official legacy Python implementation, in numbbo/coco
    #   at code-postprocessing/aRTAplots/bbobbenchmarks.py. Under
    #   numpy >= 1.16 it needs one mechanical patch:
    #   ``np.negative(idx)`` -> ``~idx`` (boolean-mask negation).

Options: ``--dims 5 10`` ``--n-points 100`` ``--seed 0``
``--rel-tol 1e-9``.
"""

import argparse
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import numpy as np


def load_legacy_python(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bbobbenchmarks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-py", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.tree_util import Partial

    import bbob_jax._src.bbob as bb
    from bbob_jax._src.transforms import lambda_func

    bn = load_legacy_python(args.legacy_py)

    def j(a: Any) -> jax.Array:
        return jnp.asarray(np.asarray(a), dtype=jnp.float64)

    def linear_slope_kw(legacy: Any) -> dict:
        dim = len(np.asarray(legacy.xopt))
        s = np.sign(np.asarray(legacy.xopt)) * 10.0 ** np.linspace(0, 1, dim)
        return {"_ls_x_opt": j(legacy.xopt), "_ls_s": j(s)}

    def scaled_rot_kw(legacy: Any, transpose: bool) -> dict:
        # F9/F19: legacy folds the scale into linearTF; bbob-jax
        # applies the scale itself and adds an (optional) shift.
        dim = np.asarray(legacy.linearTF).shape[0]
        scale = max(1, dim**0.5 / 8.0)
        mat = j(legacy.linearTF)
        return {
            "x_opt": jnp.zeros(dim, dtype=jnp.float64),
            "R": (mat.T if transpose else mat) / scale,
        }

    def schwefel_kw(legacy: Any) -> dict:
        dim = len(np.asarray(legacy.xopt))
        return {
            "_sw_ones": j(np.sign(np.asarray(legacy.xopt))),
            "_sw_x_opt_shape": j(legacy.xopt),
            "_sw_lamb": lambda_func(dim, 10.0),
        }

    def gallagher_kw(legacy: Any) -> dict:
        return {
            "R": j(legacy.rotation).T,
            "_gal_w": j(legacy.peakvalues),
            "_gal_y_rot": j(legacy.xlocal),
            "_gal_c_diags": j(legacy.arrscales),
        }

    def mat_kw(legacy: Any) -> dict:
        return {"_mat": j(legacy.linearTF).T}

    def rot_kw(legacy: Any) -> dict:
        return {"R": j(legacy.rotation).T}

    def rot_mat_kw(legacy: Any) -> dict:
        return {"R": j(legacy.rotation).T, "_mat": j(legacy.linearTF).T}

    setups: dict[int, tuple[str, Callable[[Any], dict]]] = {
        1: ("sphere", lambda legacy: {}),
        2: ("ellipsoid_seperable", lambda legacy: {}),
        3: ("rastrigin_seperable", lambda legacy: {}),
        4: ("skew_rastrigin_bueche", lambda legacy: {}),
        5: ("linear_slope", linear_slope_kw),
        6: ("attractive_sector", mat_kw),
        7: (
            "step_ellipsoid",
            lambda legacy: {
                "_mat": j(legacy.linearTF).T,
                "Q": j(legacy.rotation).T,
            },
        ),
        8: ("rosenbrock", lambda legacy: {}),
        # bbob-jax uses the row-convention (x - x_opt) @ R here
        10: ("ellipsoid", lambda legacy: {"R": j(legacy.rotation)}),
        9: ("rosenbrock_rotated", lambda legacy: scaled_rot_kw(legacy, False)),
        11: ("discuss", rot_kw),
        12: ("bent_cigar", rot_kw),
        13: ("sharp_ridge", mat_kw),
        14: ("sum_of_different_powers", rot_kw),
        15: ("rastrigin", rot_mat_kw),
        16: ("weierstrass", rot_mat_kw),
        17: ("schaffer_f7_condition_10", rot_mat_kw),
        18: ("schaffer_f7_condition_1000", rot_mat_kw),
        19: (
            "griewank_rosenbrock_f8f2",
            lambda legacy: scaled_rot_kw(legacy, True),
        ),
        20: ("schwefel_xsinx", schwefel_kw),
        21: ("gallagher_101_peaks", gallagher_kw),
        22: ("gallagher_21_peaks", gallagher_kw),
        23: ("katsuura", mat_kw),
        24: (
            "lunacek_bi_rastrigin",
            lambda legacy: {
                "_mat": j(legacy.linearTF).T,
                "_x_opt_shape": j(legacy.xopt),
            },
        ),
    }

    worst_overall = 0.0
    failures = []
    print(f"{'fid':>4} {'function':<28} {'max rel dev':>12}")
    for fid in sorted(setups):
        name, binder = setups[fid]
        worst = 0.0
        for dim in args.dims:
            rng = np.random.default_rng(args.seed + fid * 100 + dim)
            pts = rng.uniform(-5, 5, size=(args.n_points, dim))
            legacy = getattr(bn, f"F{fid}")(1)
            _, ftrue_py = legacy._evalfull(pts.copy())
            eye = jnp.eye(dim, dtype=jnp.float64)
            kw = {
                "x_opt": j(legacy.xopt),
                "f_opt": j(legacy.fopt),
                "R": eye,
                "Q": eye,
            }
            kw.update(binder(legacy))
            fn = Partial(getattr(bb, name), **kw)
            vals = np.array([float(fn(j(p))) for p in pts])
            scale = np.maximum(np.abs(ftrue_py), 1.0)
            worst = max(worst, float(np.max(np.abs(vals - ftrue_py) / scale)))
        worst_overall = max(worst_overall, worst)
        status = "ok" if worst <= args.rel_tol else "FAIL"
        if status == "FAIL":
            failures.append(fid)
        print(f"F{fid:<3} {name:<28} {worst:>12.2e}  {status}")
    print(f"\nworst overall rel dev: {worst_overall:.3e}")
    if failures:
        sys.exit(f"FAILURES: {failures}")
    print("all comparisons within tolerance")


if __name__ == "__main__":
    main()
