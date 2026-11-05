"""Cross-check of the CEC 2017 suite against the official reference code.

bbob-jax draws instance parameters from JAX keys, so the test injects
the official data files instead: for every function and dimension it
loads the official shift vectors, rotation matrices and shuffle
permutations from ``input_data/``, binds them into the bbob-jax
implementations through the keyword slots the factories normally fill,
and compares point for point in float64 with the compiled official
``cec17_test_func.cpp`` (minus its 100*i bias).

Coverage: all 29 functions, D in {10, 30}, 32 random points in
[-100, 100]^D each. The tolerance is 2e-4 absolute plus 1e-9 relative
to the largest reference value. The absolute part covers the documented
epsilon guards in the Schaffer/HGBat/HappyCat-type kernels (about 1e-6,
up to about 1e-4 after composition scaling; see ``composition.py``).
Needs the reference (see ``tests/_references.py``) and skips without it.
"""

import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial

from bbob_jax._src import cec2017
from tests._references import CEC2017, build, compiler, reference_path

FUNC_NUMS = [1] + list(range(3, 31))
DIMS = (10, 30)
N_POINTS = 32
ATOL = 2e-4
RTOL = 1e-9
HYBRIDS = set(range(11, 21))
COMPONENT_COUNTS = {21: 3, 22: 3, 23: 4, 24: 4, 25: 5, 26: 5, 27: 6, 28: 6}
SHUFFLED_COMPOSITIONS = {29: 3, 30: 3}

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


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> tuple[Path, Path]:
    """(reference directory, compiled driver)."""
    ref_dir = reference_path(CEC2017)
    gxx = compiler("g++")
    build_dir = tmp_path_factory.mktemp("cec17_ref")
    driver = build_dir / "driver.cpp"
    driver.write_text(DRIVER_C)
    exe = build_dir / "cec17_ref"
    src = ref_dir / "cec17_test_func.cpp"
    build([gxx, "-O2", "-o", str(exe), str(driver), str(src), "-lm"])
    return ref_dir, exe


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _row(line: str, n: int) -> np.ndarray:
    return np.array(line.split(), dtype=float)[:n]


def _load_official(
    data_dir: Path, func: int, nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load (shift, matrices, shuffle) as the reference code does."""
    m_vals = np.loadtxt(data_dir / f"M_{func}_D{nx}.txt").ravel()
    lines = (data_dir / f"shift_data_{func}.txt").read_text().splitlines()
    if func <= 20:  # simple functions and hybrids: one matrix, one shift
        mat = m_vals[: nx * nx].reshape(nx, nx)
        shift = _row(lines[0], nx)
    else:
        n_comp = COMPONENT_COUNTS.get(func) or SHUFFLED_COMPOSITIONS[func]
        mat = m_vals[: n_comp * nx * nx].reshape(n_comp, nx, nx)
        shift = np.stack([_row(lines[i], nx) for i in range(n_comp)])

    shuffle = None
    if func in HYBRIDS or func in SHUFFLED_COMPOSITIONS:
        raw = np.loadtxt(data_dir / f"shuffle_data_{func}_D{nx}.txt")
        raw = raw.astype(int).ravel() - 1
        if func in HYBRIDS:
            shuffle = raw[:nx]
        else:
            n_comp = SHUFFLED_COMPOSITIONS[func]
            shuffle = raw[: n_comp * nx].reshape(n_comp, nx)
    return shift, mat, shuffle


@pytest.mark.parametrize("nx", DIMS, ids=lambda nx: f"D{nx}")
@pytest.mark.parametrize("func", FUNC_NUMS, ids=lambda f: f"f{f}")
def test_matches_official_reference(func, nx, reference):
    ref_dir, exe = reference
    rng = np.random.default_rng([func, nx])
    x = rng.uniform(-100.0, 100.0, size=(N_POINTS, nx))
    proc = subprocess.run(
        [str(exe), str(func), str(nx), str(N_POINTS)],
        input=" ".join(f"{v:.17g}" for v in x.ravel()),
        capture_output=True,
        text=True,
        cwd=ref_dir,  # the reference reads input_data/ relative to cwd
        check=True,
    )
    f_ref = np.array([float(v) for v in proc.stdout.split()]) - 100.0 * func

    shift, mat, shuffle = _load_official(ref_dir / "input_data", func, nx)
    kw = {
        "x_opt": jnp.asarray(shift),
        "f_opt": jnp.asarray(0.0),
        "R": jnp.asarray(mat),
        "Q": jnp.zeros_like(jnp.asarray(mat)),
    }
    if shuffle is not None:
        kw["_shuffle"] = jnp.asarray(shuffle)
    fn = Partial(getattr(cec2017, f"f{func}"), **kw)
    f_jax = np.array(jax.vmap(fn)(jnp.asarray(x, dtype=jnp.float64)))

    abs_diff = float(np.max(np.abs(f_jax - f_ref)))
    tol = ATOL + RTOL * float(np.max(np.abs(f_ref)))
    print(f"f{func} D={nx}: max |diff| {abs_diff:.2e} (tolerance {tol:.2e})")
    assert abs_diff <= tol, f"f{func} D={nx}: {abs_diff:.2e} > {tol:.2e}"
