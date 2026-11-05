"""Cross-check of the CEC 2013 LSGO suite against the official C++ code.

LSGO's parameters are the official fixed constants, so no injection is
needed: the test evaluates bbob-jax and the official competition C++
code on the same random points in each function's box, at the native
dimension, in float64. The C++ code is the one Daniel Molina's
``cec2013lsgo`` 2.2 package wraps; the test compiles its sources
directly with a small driver, because the 2018 package itself does not
build on current Python and setuptools.

Coverage: all 15 functions, 5 points each, relative tolerance 1e-6 (the
same as the golden-pin test in ``test_cec2013lsgo.py``). Every function
but F7 agrees to about 1e-12.

F7 is a known deviation. bbob-jax, like MetaBox (the source of the port
and of the golden pins), applies T_osz and T_asy to F7's separable
remainder before the sphere; the official C++ uses a plain sphere
there. On the points checked the relative difference ranges from
about 1e-3 to 10. The test is marked as an expected failure, strict, so
it fails once F7 is changed to match. Needs the reference (see
``tests/_references.py``) and skips without it.
"""

import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import bbob_jax as B
from tests._references import CEC2013LSGO, build, compiler, reference_path

FIDS = list(range(1, 16))
N_POINTS = 5
RTOL = 1e-6
REFERENCE_DIM = 1000  # the C++ code always reads 1000 coordinates

DRIVER_CPP = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "eval_func.h"
int main(int argc, char **argv) {
    if (argc != 4) return 1;
    int fid = atoi(argv[1]);
    int npoints = atoi(argv[2]);
    set_func(fid);
    set_data_dir(argv[3]);
    std::vector<double> x(1000);
    for (int p = 0; p < npoints; p++) {
        for (int i = 0; i < 1000; i++)
            if (scanf("%lf", &x[i]) != 1) return 2;
        printf("%.17g\n", eval_sol(x.data()));
    }
    free_func();
    return 0;
}
"""

F7_REMAINDER = pytest.mark.xfail(
    strict=True,
    # only the value mismatch is expected; a missing reference or a
    # crashing driver must still fail
    raises=AssertionError,
    reason=(
        "known deviation: bbob-jax (like MetaBox) applies T_osz/T_asy to "
        "F7's separable remainder; the official C++ uses a plain sphere"
    ),
)


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> tuple[Path, Path]:
    """(compiled driver, data directory)."""
    src = reference_path(CEC2013LSGO)
    gxx = compiler("g++")
    build_dir = tmp_path_factory.mktemp("cec2013lsgo_ref")
    driver = build_dir / "driver.cpp"
    driver.write_text(DRIVER_CPP)
    exe = build_dir / "cec2013lsgo_ref"
    sources = ["eval_func.cpp", "Benchmarks.cpp"]
    sources += [f"F{fid}.cpp" for fid in FIDS]
    build(
        [gxx, "-O2", "-std=c++11", "-I", str(src), "-o", str(exe)]
        + [str(driver)]
        + [str(src / s) for s in sources]
        + ["-lm"]
    )
    return exe, src / "cdatafiles"


@pytest.mark.parametrize(
    "fid",
    [
        pytest.param(fid, marks=F7_REMAINDER) if fid == 7 else fid
        for fid in FIDS
    ],
    ids=lambda fid: f"f{fid}",
)
def test_matches_official_cpp(fid, reference):
    exe, data_dir = reference
    name = f"cec2013lsgo_f{fid}"
    ndim = 905 if fid in (13, 14) else 1000
    lo, hi = B.cec2013lsgo_bounds[name]
    xs = np.random.default_rng(fid).uniform(lo, hi, size=(N_POINTS, ndim))

    # the overlapping F13/F14 ignore the coordinates past their 905
    padded = np.zeros((N_POINTS, REFERENCE_DIM))
    padded[:, :ndim] = xs
    proc = subprocess.run(
        [str(exe), str(fid), str(N_POINTS), str(data_dir)],
        input=" ".join(f"{v:.17g}" for v in padded.ravel()),
        capture_output=True,
        text=True,
        check=True,
    )
    f_ref = np.array([float(v) for v in proc.stdout.split()])

    with jax.enable_x64(True):
        fn, _ = B.cec2013lsgo_registry[name](ndim=ndim, key=jr.key(0))
        f_jax = np.array([float(fn(jnp.asarray(x))) for x in xs])

    rel = float(np.max(np.abs(f_jax - f_ref) / np.abs(f_ref)))
    print(f"F{fid}: max relative deviation {rel:.2e}")
    assert rel <= RTOL, f"F{fid}: {rel:.2e} > {RTOL:.0e}"
