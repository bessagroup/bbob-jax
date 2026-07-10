"""Cross-validate the CEC 2017 suite against the official reference code.

One-off development validation (not part of CI — instances in bbob-jax are
seed-generated, so this script instead *injects the official data files*
into the bbob-jax function definitions and compares point-for-point against
the compiled official ``cec17_test_func.cpp``):

1. Compiles the official source with a tiny driver (needs ``gcc``/``g++``).
2. For every function and dimension, loads the official shift vectors,
   rotation matrices and shuffle permutations from ``input_data/`` and binds
   them into the bbob-jax implementations via ``jax.tree_util.Partial`` —
   exactly the parameter slots the factories normally fill with sampled
   values.
3. Evaluates both on random points in ``[-100, 100]^D`` (float64) and
   reports the max absolute/relative deviation per function.

Expected deviations are the documented epsilon guards (~1e-6 absolute on
Schaffer/HGBat/HappyCat-type kernels, up to ~1e-4 after composition lambda
scaling) — see the kernel docstrings in ``composition.py``.

Usage::

    uv run python scripts/crosscheck_cec2017.py --ref-dir /path/to/CEC17_fast_pow
    # where ref-dir contains cec17_test_func.cpp and input_data/
    # (from the official P-N-Suganthan/CEC2017-BoundContrained repo,
    #  CEC17_fast_pow-C++.zip)

Options: ``--dims 10 30`` ``--n-points 32`` ``--seed 0``.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DRIVER_C = r"""
#include <stdio.h>
#include <stdlib.h>
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;
void cec17_test_func(double *, double *, int, int, int);
int main(int argc, char **argv)
{
    int func = atoi(argv[1]), nx = atoi(argv[2]), mx = atoi(argv[3]);
    double *x = (double *)malloc(sizeof(double) * nx * mx);
    double *f = (double *)malloc(sizeof(double) * mx);
    for (int i = 0; i < nx * mx; i++)
        if (scanf("%lf", &x[i]) != 1) return 2;
    cec17_test_func(x, f, nx, mx, func);
    for (int i = 0; i < mx; i++)
        printf("%.17g\n", f[i]);
    return 0;
}
"""

FUNC_NUMS = [1] + list(range(3, 31))
HYBRIDS = set(range(11, 21))
COMPOSITIONS = set(range(21, 31))
COMPONENT_COUNTS = {21: 3, 22: 3, 23: 4, 24: 4, 25: 5, 26: 5, 27: 6, 28: 6}
SHUFFLED_COMPOSITIONS = {29: 3, 30: 3}


def compile_reference(ref_dir: Path, build_dir: Path) -> Path:
    src = ref_dir / "cec17_test_func.cpp"
    if not src.exists():
        src = ref_dir / "cec17_test_func.c"
    if not src.exists():
        sys.exit(f"no cec17_test_func.c(pp) in {ref_dir}")
    driver = build_dir / "driver.c"
    driver.write_text(DRIVER_C)
    exe = build_dir / "cec17_ref"
    # gcc handles the .cpp fine here (it is C in a .cpp file), but use g++
    # for the .cpp flavor to be safe about the extern declarations.
    compiler = "g++" if src.suffix == ".cpp" else "gcc"
    subprocess.run(
        [compiler, "-O2", "-o", str(exe), str(driver), str(src), "-lm"],
        check=True,
    )
    return exe


def load_official(data_dir: Path, func: int, nx: int):
    """Load (shift, matrices, shuffle) as the reference code does."""
    m_vals = np.loadtxt(data_dir / f"M_{func}_D{nx}.txt").ravel()
    shift_lines = (
        (data_dir / f"shift_data_{func}.txt").read_text().splitlines()
    )

    if func < 20 and func not in COMPOSITIONS:
        mat = m_vals[: nx * nx].reshape(nx, nx)
        shift = np.fromstring(shift_lines[0], sep=" ")[:nx]
    elif func in HYBRIDS or func < 21:  # func == 20
        mat = m_vals[: nx * nx].reshape(nx, nx)
        shift = np.fromstring(shift_lines[0], sep=" ")[:nx]
    else:
        n_comp = COMPONENT_COUNTS.get(func) or SHUFFLED_COMPOSITIONS[func]
        mat = m_vals[: n_comp * nx * nx].reshape(n_comp, nx, nx)
        shift = np.stack(
            [
                np.fromstring(shift_lines[i], sep=" ")[:nx]
                for i in range(n_comp)
            ]
        )

    shuffle = None
    if func in HYBRIDS:
        shuffle = (
            np.loadtxt(data_dir / f"shuffle_data_{func}_D{nx}.txt")
            .astype(int)
            .ravel()[:nx]
            - 1
        )
    elif func in SHUFFLED_COMPOSITIONS:
        n_comp = SHUFFLED_COMPOSITIONS[func]
        raw = (
            np.loadtxt(data_dir / f"shuffle_data_{func}_D{nx}.txt")
            .astype(int)
            .ravel()
        )
        shuffle = raw[: n_comp * nx].reshape(n_comp, nx) - 1
    return shift, mat, shuffle


def build_bbob_jax_fn(func: int, shift, mat, shuffle):
    import jax.numpy as jnp
    from jax.tree_util import Partial

    from bbob_jax._src import cec2017

    fn = getattr(cec2017, f"f{func}")
    zero = jnp.asarray(0.0)
    kwargs = {
        "x_opt": jnp.asarray(shift),
        "f_opt": zero,
        "R": jnp.asarray(mat),
        "Q": jnp.zeros_like(jnp.asarray(mat)),
    }
    if shuffle is not None:
        kwargs["_shuffle"] = jnp.asarray(shuffle)
    return Partial(fn, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--n-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    data_dir = args.ref_dir / "input_data"
    rng = np.random.default_rng(args.seed)
    failures = []

    with tempfile.TemporaryDirectory() as build:
        exe = compile_reference(args.ref_dir, Path(build))
        for nx in args.dims:
            for func in FUNC_NUMS:
                x = rng.uniform(-100.0, 100.0, size=(args.n_points, nx))
                proc = subprocess.run(
                    [str(exe), str(func), str(nx), str(args.n_points)],
                    input=" ".join(f"{v:.17g}" for v in x.ravel()),
                    capture_output=True,
                    text=True,
                    cwd=args.ref_dir,
                    check=True,
                )
                ref = (
                    np.array([float(line) for line in proc.stdout.split()])
                    - 100.0 * func
                )

                shift, mat, shuffle = load_official(data_dir, func, nx)
                ours_fn = build_bbob_jax_fn(func, shift, mat, shuffle)
                ours = np.array(
                    jax.vmap(ours_fn)(jnp.asarray(x, dtype=jnp.float64))
                )

                abs_diff = np.max(np.abs(ours - ref))
                rel_diff = np.max(
                    np.abs(ours - ref) / np.maximum(np.abs(ref), 1.0)
                )
                tol = args.atol + args.rtol * np.max(np.abs(ref))
                status = "OK " if abs_diff <= tol else "FAIL"
                if abs_diff > tol:
                    failures.append((func, nx, abs_diff))
                print(
                    f"f{func:<3d} D={nx:<4d} max|diff|={abs_diff:12.4e}  "
                    f"max rel={rel_diff:12.4e}  {status}"
                )

    if failures:
        print(f"\n{len(failures)} function/dim pairs beyond tolerance")
        return 1
    print("\nall functions match the official reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
