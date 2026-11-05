"""Cross-check of the CEC 2005 compositions against the official C code.

Covers what bbob-jax takes from the official code for F15-F25: the
composition weights (``calc_weight`` in ``src/def2.c``: every weight but
the largest damped by ``1 - w_max**10``, then all divided by their sum),
the height normalization (``calc_benchmark_norm``, computed once), and
the noise of F17 (``1 + 0.2*|N|`` on the whole value) and F24/F25
(``1 + 0.1*|N|`` on the sphere component, ``src/def4.c``).

bbob-jax draws its instances from JAX keys, so the test injects them
into the official code: it builds ``def1-4.c`` and ``aux.c`` with a
small driver that runs the official ``initialize()`` (which sets each
function's sigma, lambda and biases), then overwrites the component
optima, the rotation matrices and the global bias with those of the
bbob-jax instance, and calls ``calc_benchmark_norm()``. The official
noise generator is replaced by a stub that returns the draw bbob-jax
takes from the same key, ``jax.random.normal(key)``; the stub returns 0
while the normalization is computed, because bbob-jax normalizes F24/F25
noise-free where the official code draws the noise once there.

* F15-F22: full values, point for point. F15 is unrotated: it
  evaluates, and computes its height normalization, with identity
  matrices, as the official code does.
* F24, F25: the noise's contribution, ``fn(x, key) - fn_true(x)``
  against the official value with the draw minus the one with draw 0.
  Their full values differ by design: the non-continuous Scaffer and
  Rastrigin components are soft-rounded in bbob-jax (see
  ``cec2005_deviations``), and that part cancels in the difference.
* F23 is left out: its input is rounded, softly in bbob-jax. Its
  weights are F21's, which are checked.

The two files bbob-jax cannot use as published are patched in the build
directory, not on disk: ``global.h`` selects F11 with a ``# define``
that clashes with the function chosen on the compiler command line, and
``rand.h`` includes the SPRNG library the tarball ships prebuilt for
i686. Needs the reference (see ``tests/_references.py``) and gcc.
"""

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import bbob_jax
from tests._references import CEC2005, build, compiler, reference_path

DIMS = (10, 30)
N_POINTS = 24
ATOL = 1e-8
RTOL = 1e-10
FULL_VALUE = (15, 16, 17, 18, 19, 20, 21, 22)
NOISE_ONLY = (24, 25)
SOURCES = ("aux.c", "def1.c", "def2.c", "def3.c", "def4.c")

STUB_RAND_H = r"""
#ifndef _RAND_H
#define _RAND_H
#include "global.h"
extern long double stub_deviate;
long double randomnormaldeviate();
#endif
"""

DRIVER_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "sub.h"
#include "rand.h"

long double stub_deviate = 0.0;
long double randomnormaldeviate() { return stub_deviate; }

int main(void)
{
    int i, j, k, n_points;
    long double *x, value;
    nfunc = 10;
    if (scanf("%d", &nreal) != 1) return 2;
    allocate_memory();
    initialize();  /* official sigma, lambda, bias; reads input_data/ */
    for (i = 0; i < nfunc; i++)
        for (j = 0; j < nreal; j++)
            if (scanf("%Lf", &o[i][j]) != 1) return 2;
    /* bbob-jax maps z = M (x - o) / lambda; the official code
       z[k] = sum_j l[j][k] (x - o)[j] / lambda, so l = M transposed. */
    for (i = 0; i < nfunc; i++)
        for (k = 0; k < nreal; k++)
            for (j = 0; j < nreal; j++)
                if (scanf("%Lf", &l[i][j][k]) != 1) return 2;
    if (scanf("%Lf", &global_bias) != 1) return 2;
    stub_deviate = 0.0;
    calc_benchmark_norm();
    if (scanf("%d", &n_points) != 1) return 2;
    x = (long double *)malloc(nreal * sizeof(long double));
    for (i = 0; i < n_points; i++)
    {
        for (j = 0; j < nreal; j++)
            if (scanf("%Lf", &x[j]) != 1) return 2;
        if (scanf("%Lf", &stub_deviate) != 1) return 2;
        value = calc_benchmark_func(x);
        printf("RESULT %.21Le\n", value);
    }
    free(x);
    free_memory();
    return 0;
}
"""


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> tuple[Path, dict[int, Path]]:
    """(reference directory, compiled driver per function number)."""
    ref_dir = reference_path(CEC2005)
    gcc = compiler("gcc")
    build_dir = tmp_path_factory.mktemp("cec2005_ref")
    for name in SOURCES + ("sub.h",):
        shutil.copy(ref_dir / name, build_dir / name)
    global_h = (ref_dir / "global.h").read_text()
    assert "# define f11\n" in global_h
    (build_dir / "global.h").write_text(global_h.replace("# define f11\n", ""))
    (build_dir / "rand.h").write_text(STUB_RAND_H)
    (build_dir / "driver.c").write_text(DRIVER_C)
    sources = [str(build_dir / n) for n in SOURCES + ("driver.c",)]
    exes = {}
    for func in (15,) + FULL_VALUE + NOISE_ONLY:
        exe = build_dir / f"cec2005_f{func}"
        # global.h defines its variables; -fcommon merges the copies.
        build(
            [gcc, "-O2", "-fcommon", f"-Df{func}", "-o", str(exe)]
            + sources
            + ["-lm"]
        )
        exes[func] = exe
    return ref_dir, exes


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _official(exe, ref_dir, kw, x, deviates) -> np.ndarray:
    """Evaluate the official function on the bbob-jax instance ``kw``."""
    numbers = [str(x.shape[1])]
    numbers += [repr(float(v)) for v in np.ravel(kw["x_opt"])]
    matrices = np.asarray(kw["R"])
    if kw.get("_unrotated", False):
        matrices = np.broadcast_to(np.eye(x.shape[1]), matrices.shape)
    numbers += [repr(float(v)) for v in np.ravel(matrices)]
    numbers += [repr(float(kw["f_opt"])), str(len(x))]
    for point, deviate in zip(x, deviates):
        numbers += [repr(float(v)) for v in point] + [repr(float(deviate))]
    proc = subprocess.run(
        [str(exe)],
        input=" ".join(numbers),
        capture_output=True,
        text=True,
        cwd=ref_dir,  # initialize() reads input_data/ relative to cwd
        check=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    return np.array([float(ln.split()[1]) for ln in lines])


def _points(kw, nx, seed) -> np.ndarray:
    """Half uniform in [-5, 5]^D, half near component optima.

    Near an optimum the damping ``1 - w_max**10`` is far from 1, which
    is where the weights differ most from a plain normalization.
    """
    rng = np.random.default_rng(seed)
    x_opt = np.asarray(kw["x_opt"])
    far = rng.uniform(-5.0, 5.0, size=(N_POINTS // 2, nx))
    comps = rng.integers(0, len(x_opt), size=N_POINTS // 2)
    comps[:4] = len(x_opt) - 1  # the noisy sphere of F24/F25
    scale = rng.choice([0.01, 0.1, 0.5], size=(N_POINTS // 2, 1))
    near = x_opt[comps] + scale * rng.normal(size=(N_POINTS // 2, nx))
    return np.concatenate([far, near])


def _assert_close(got, ref, label):
    abs_diff = float(np.max(np.abs(got - ref)))
    tol = ATOL + RTOL * float(np.max(np.abs(ref)))
    print(f"{label}: max |diff| {abs_diff:.2e} (tolerance {tol:.2e})")
    assert abs_diff <= tol, f"{label}: {abs_diff:.2e} > {tol:.2e}"


@pytest.mark.parametrize("nx", DIMS, ids=lambda nx: f"D{nx}")
@pytest.mark.parametrize(
    "func",
    FULL_VALUE,
    ids=lambda f: f"f{f}",
)
def test_matches_official_reference(func, nx, reference):
    ref_dir, exes = reference
    p = bbob_jax.problem(f"f{func}", ndim=nx, key=jr.key(func))
    kw = dict(p.fn.keywords, _unrotated=func == 15)
    x = _points(kw, nx, seed=[func, nx])
    keys = jr.split(jr.key(1000 + func), len(x))
    xs = jnp.asarray(x)
    if p.noisy:
        got = np.array(jax.vmap(p.fn)(xs, keys))
        deviates = np.array(jax.vmap(jr.normal)(keys))
    else:
        got = np.array(jax.vmap(p.fn)(xs))
        deviates = np.zeros(len(x))
    ref = _official(exes[func], ref_dir, kw, x, deviates)
    _assert_close(got, ref, f"f{func} D={nx}")


@pytest.mark.parametrize("nx", DIMS, ids=lambda nx: f"D{nx}")
@pytest.mark.parametrize("func", NOISE_ONLY, ids=lambda f: f"f{f}")
def test_noise_contribution_matches_official_reference(func, nx, reference):
    ref_dir, exes = reference
    p = bbob_jax.problem(f"f{func}", ndim=nx, key=jr.key(func))
    kw = p.fn.keywords
    x = _points(kw, nx, seed=[func, nx])
    keys = jr.split(jr.key(1000 + func), len(x))
    xs = jnp.asarray(x)
    got = np.array(jax.vmap(p.fn)(xs, keys) - jax.vmap(p.fn_true)(xs))
    deviates = np.array(jax.vmap(jr.normal)(keys))
    noisy = _official(exes[func], ref_dir, kw, x, deviates)
    quiet = _official(exes[func], ref_dir, kw, x, np.zeros(len(x)))
    ref = noisy - quiet
    assert np.max(np.abs(ref)) > 1e-6  # the noise acts somewhere
    _assert_close(got, ref, f"f{func} D={nx} noise")
