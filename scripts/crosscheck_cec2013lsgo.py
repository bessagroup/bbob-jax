"""Cross-validate the CEC 2013 LSGO suite against an external reference.

One-off development validation (not part of CI). Unlike the CEC 2005/2017
crosschecks, LSGO's parameters are the official fixed constants, so this
script compares the bbob-jax port directly against a reference
implementation on random points in each function's box, at the native
dimension, in float64.

Canonical oracle: Daniel Molina's ``cec2013lsgo`` package (a C-backed
wrapper over the official competition code)::

    pip install cec2013lsgo            # needs a C toolchain; internet
    uv run --no-sync python scripts/crosscheck_cec2013lsgo.py

Fallback oracle (when ``cec2013lsgo`` cannot be installed, e.g. an offline
compute node): MetaBox's NumPy reference — the exact source the port was
derived from — loaded directly::

    uv run --no-sync python scripts/crosscheck_cec2013lsgo.py \
        --oracle metabox \
        --metabox /path/to/MetaBox/src/environment/problem

Use ``--write-golden tests/data/cec2013lsgo_golden.npz`` to (re)generate the
CI regression pins. Options: ``--n-points``, ``--seed``, ``--rtol``.

The committed golden was generated with ``--oracle metabox`` (MetaBox
``MetaEvo/MetaBox@5565a28``); re-run with the default ``dmolina`` oracle on
a networked machine to confirm against the C-backed official code.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Third-party
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

N_FUNCTIONS = 15
OVERLAP_DIM_IDS = (13, 14)


def native_dim(fid: int) -> int:
    """Native dimension of LSGO function ``fid`` (905 for F13/F14)."""
    return 905 if fid in OVERLAP_DIM_IDS else 1000


def _bounds(fid: int) -> tuple[float, float]:
    """Search-space box bounds for LSGO function ``fid``."""
    import bbob_jax as B

    return B.cec2013lsgo_bounds[f"cec2013lsgo_f{fid}"]


def make_dmolina_oracle() -> Callable[[int, np.ndarray], float]:
    """Return ``eval_fn(fid, x) -> float`` backed by ``cec2013lsgo``."""
    from cec2013lsgo.cec2013 import Benchmark

    bench = Benchmark()
    funcs = {fid: bench.get_function(fid) for fid in range(1, N_FUNCTIONS + 1)}

    def eval_fn(fid: int, x: np.ndarray) -> float:
        # dmolina functions expect the full 1000-D input; the overlapping
        # functions ignore coordinates past their 905 used indices.
        if fid in OVERLAP_DIM_IDS and x.shape[0] == 905:
            x = np.concatenate([x, np.zeros(1000 - 905)])
        return float(funcs[fid](x))

    return eval_fn


def make_metabox_oracle(
    metabox_problem_dir: str,
) -> Callable[[int, np.ndarray], float]:
    """Return ``eval_fn(fid, x)`` backed by MetaBox's NumPy reference.

    Loads ``cec2013lsgo_numpy`` directly (torch stubbed, minimal package
    skeleton) so no torch / gym dependencies are pulled in.
    """
    torch_stub: Any = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})  # satisfy jaxtyping's probe
    sys.modules.setdefault("torch", torch_stub)

    def load_file(
        name: str, path: str, package: str | None = None
    ) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        if package is not None:
            mod.__package__ = package
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    for pkg in (
        "environment",
        "environment.problem",
        "environment.problem.SOO",
        "environment.problem.SOO.CEC2013LSGO",
    ):
        skeleton: Any = types.ModuleType(pkg)
        skeleton.__path__ = []
        sys.modules[pkg] = skeleton

    base = Path(metabox_problem_dir)
    bp = load_file(
        "environment.problem.basic_problem", str(base / "basic_problem.py")
    )
    ep_mod: Any = sys.modules["environment.problem"]
    ep_mod.basic_problem = bp
    ref = load_file(
        "environment.problem.SOO.CEC2013LSGO.cec2013lsgo_numpy",
        str(base / "SOO/CEC2013LSGO/cec2013lsgo_numpy.py"),
        package="environment.problem.SOO.CEC2013LSGO",
    )
    problems = {fid: getattr(ref, f"F{fid}")() for fid in range(1, 16)}

    def eval_fn(fid: int, x: np.ndarray) -> float:
        if fid in OVERLAP_DIM_IDS and x.shape[0] == 905:
            x = np.concatenate([x, np.zeros(1000 - 905)])
        y = np.asarray(problems[fid].func(x.reshape(1, -1)))
        return float(y.ravel()[0])

    return eval_fn


def main() -> None:
    """Run the cross-check and optionally write the golden regression pins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle", choices=("dmolina", "metabox"), default="dmolina"
    )
    parser.add_argument(
        "--metabox",
        type=str,
        default=None,
        help="MetaBox .../environment/problem dir",
    )
    parser.add_argument("--n-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--write-golden", type=str, default=None)
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr

    import bbob_jax as B

    if args.oracle == "dmolina":
        oracle = make_dmolina_oracle()
    else:
        if args.metabox is None:
            raise SystemExit("--oracle metabox requires --metabox <dir>")
        oracle = make_metabox_oracle(args.metabox)

    rng = np.random.RandomState(args.seed)
    golden: dict[str, np.ndarray] = {}
    worst = 0.0
    for fid in range(1, N_FUNCTIONS + 1):
        ndim = native_dim(fid)
        lo, hi = _bounds(fid)
        fn, _ = B.cec2013lsgo_registry[f"cec2013lsgo_f{fid}"](
            ndim=ndim, key=jr.key(0)
        )
        xs = rng.uniform(lo, hi, size=(args.n_points, ndim))
        refs = np.array([oracle(fid, x) for x in xs])
        jaxs = np.array([float(fn(jnp.asarray(x))) for x in xs])
        rel = np.max(np.abs(jaxs - refs) / np.maximum(np.abs(refs), 1e-12))
        worst = max(worst, rel)
        flag = "OK " if rel < args.rtol else "!!!"
        print(f"{flag} cec2013lsgo_f{fid:<2}: max rel dev = {rel:.3e}")
        golden[f"x_{fid}"] = xs
        golden[f"f_{fid}"] = refs

    print(
        f"\nworst relative deviation: {worst:.3e} (tolerance {args.rtol:.0e})"
    )

    if args.write_golden:
        out = Path(args.write_golden)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **golden)  # type: ignore[arg-type]
        print(f"wrote golden pins ({args.oracle}) to {out}")


if __name__ == "__main__":
    main()
