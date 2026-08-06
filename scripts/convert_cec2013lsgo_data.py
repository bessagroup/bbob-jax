"""Convert the CEC 2013 LSGO reference constants to ``.npz`` package data.

The CEC 2013 Large-Scale Global Optimization suite is *defined* by a
set of official constants (shift vectors, subcomponent rotation
matrices, permutations, subcomponent sizes and weights) that cannot be
regenerated from a seed the way ``bbob-jax``'s CEC 2005/2017 parameters
are. These constants are vendored into the package as ``.npz`` files so
that the suite is self-contained and reproducible off-cluster.

This script reads the plain-text datafiles shipped by MetaBox (itself a
copy of Daniel Molina's ``cec2013lsgo`` reference implementation, which
mirrors the official C/MATLAB code) and writes one compressed ``.npz``
per function into ``src/bbob_jax/_src/cec2013lsgo_data/``.

Datafile provenance (see ``cec2013lsgo_data/PROVENANCE.md``):
  MetaEvo/MetaBox @ 5565a28  src/environment/problem/SOO/CEC2013LSGO/datafile
  upstream: https://github.com/dmolina/cec2013lsgo
  benchmark: Li, Tang, Omidvar, Yang, Qin (2013).

Text formats
------------
``F{i}-xopt.txt``   : one float per line              -> ``xopt``  (D,)
``F{i}-R{k}.txt``   : k lines of k comma-separated     -> ``R{k}``  (k, k)
``F{i}-p.txt``      : one line, comma-separated ints    -> ``p``     (D,)
``F{i}-s.txt``      : one int per line                   -> ``s``     (m,)
``F{i}-w.txt``      : one float per line                 -> ``w``     (m,)

Only ``xopt`` is present for every function; the fully-separable and
single-function cases (F1, F2, F3, F12, F15) ship ``xopt`` alone.

Usage
-----
    uv run --no-sync python scripts/convert_cec2013lsgo_data.py \
        [--src /path/to/MetaBox/.../CEC2013LSGO/datafile]

Values are stored as ``float64`` (rotations, xopt, weights) and ``int32``
(permutation, sizes); the runtime loader downcasts to the configured JAX
precision. The permutation is stored verbatim (1-indexed, as in the
reference) and de-indexed in the function implementation.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import argparse
from pathlib import Path

# Third-party
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = [
    "Martin van der Schelling",
    "MetaEvolution Lab",
    "Daniel Molina",
]
__status__ = "Stable"
# =============================================================================

DEFAULT_SRC = Path(
    "/home/mpvanderschell/MetaBox/src/environment/problem/"
    "SOO/CEC2013LSGO/datafile"
)
OUT_DIR = Path(__file__).resolve().parents[1] / (
    "src/bbob_jax/_src/cec2013lsgo_data"
)
N_FUNCTIONS = 15
ROT_SIZES = (25, 50, 100)


def _load_vector(path: Path) -> np.ndarray:
    """Read a one-value-per-line text file into a 1-D array."""
    return np.loadtxt(path, dtype=np.float64, ndmin=1)


def _load_matrix(path: Path) -> np.ndarray:
    """Read a comma-separated k-by-k text file into a 2-D array."""
    return np.loadtxt(path, dtype=np.float64, delimiter=",", ndmin=2)


def _load_perm(path: Path) -> np.ndarray:
    """Read the single comma-separated permutation line into a 1-D array."""
    return np.loadtxt(path, dtype=np.int32, delimiter=",", ndmin=1)


def convert_function(src: Path, fid: int) -> dict[str, np.ndarray]:
    """Collect every constant present for function ``fid``."""
    arrays: dict[str, np.ndarray] = {}

    xopt_path = src / f"F{fid}-xopt.txt"
    if not xopt_path.exists():
        raise FileNotFoundError(f"missing required shift vector: {xopt_path}")
    arrays["xopt"] = _load_vector(xopt_path)

    for k in ROT_SIZES:
        rot_path = src / f"F{fid}-R{k}.txt"
        if rot_path.exists():
            mat = _load_matrix(rot_path)
            if mat.shape != (k, k):
                raise ValueError(
                    f"F{fid} R{k}: expected ({k}, {k}), got {mat.shape}"
                )
            arrays[f"R{k}"] = mat

    p_path = src / f"F{fid}-p.txt"
    if p_path.exists():
        arrays["p"] = _load_perm(p_path)

    s_path = src / f"F{fid}-s.txt"
    if s_path.exists():
        arrays["s"] = _load_vector(s_path).astype(np.int32)

    w_path = src / f"F{fid}-w.txt"
    if w_path.exists():
        arrays["w"] = _load_vector(w_path)

    return arrays


def main() -> None:
    """Convert all 15 functions' datafiles to ``.npz`` package data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"source datafile directory not found: {args.src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_doc = '"""Vendored CEC 2013 LSGO constants (see PROVENANCE.md)."""\n'
    (OUT_DIR / "__init__.py").write_text(init_doc)

    for fid in range(1, N_FUNCTIONS + 1):
        arrays = convert_function(args.src, fid)
        out_path = OUT_DIR / f"F{fid}.npz"
        np.savez_compressed(out_path, **arrays)
        keys = ", ".join(sorted(arrays))
        dim = arrays["xopt"].shape[0]
        size_kb = out_path.stat().st_size / 1024
        print(f"F{fid:>2}: D={dim:>4}  [{keys}]  -> {out_path.name} "
              f"({size_kb:.0f} KiB)")

    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.npz")) / 1024
    print(f"\nWrote {N_FUNCTIONS} files to {OUT_DIR} ({total:.0f} KiB total)")


if __name__ == "__main__":
    main()
