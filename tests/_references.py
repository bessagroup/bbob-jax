"""Find and build the reference code the cross-check tests compare against.

The cross-check tests (``test_crosscheck_*.py``) compare bbob-jax point
for point with the official reference implementations. That code is not
in the repository: ``scripts/fetch_references.sh DIR`` downloads pinned,
checksummed copies into ``DIR``, and the tests find ``DIR`` through the
``BBOB_JAX_REFERENCES`` environment variable.

Without the variable, or without the file a test needs, the test skips
and says why. With ``BBOB_JAX_REQUIRE_REFERENCES=1``, as CI sets it, a
missing reference or compiler fails the test instead, so CI cannot pass
by skipping.
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import numpy as np
import pytest

ENV_DIR = "BBOB_JAX_REFERENCES"
ENV_REQUIRE = "BBOB_JAX_REQUIRE_REFERENCES"

# Layout of the directory scripts/fetch_references.sh fills.
LEGACY_PY = "bbobbenchmarks.py"  # COCO's legacy Python BBOB code
LEGACY_C = "bbob-legacy-c"  # legacy BBOB C code (noisy suite)
CEC2017 = "CEC17_fast_pow"  # official CEC 2017 C++ code + input_data/
CEC2013LSGO = "cec2013lsgo-2.2/cec2013lsgo"  # official LSGO C++ code
CEC2005 = "cec2005-c/src"  # official CEC 2005 C code + input_data/


def missing(reason: str) -> NoReturn:
    """Skip the test, or fail it when references are required."""
    if os.environ.get(ENV_REQUIRE) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def reference_path(name: str) -> Path:
    """Path of one reference inside ``$BBOB_JAX_REFERENCES``."""
    root = os.environ.get(ENV_DIR)
    if not root:
        missing(
            f"{ENV_DIR} is not set: fill a directory with "
            f"scripts/fetch_references.sh and point {ENV_DIR} at it"
        )
    path = Path(root) / name
    if not path.exists():
        missing(
            f"reference {path} not found: run "
            f"scripts/fetch_references.sh {root}"
        )
    return path


def compiler(name: str) -> str:
    """Full path of a C/C++ compiler, or skip/fail when it is absent."""
    exe = shutil.which(name)
    if exe is None:
        missing(f"{name} not found on PATH; it builds the reference code")
    return exe


def build(command: list[str]) -> None:
    """Compile reference code, showing the compiler output on failure."""
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"building the reference failed:\n{proc.stderr}")


def load_legacy_bbob(path: Path) -> ModuleType:
    """Import COCO's legacy ``bbobbenchmarks.py``.

    One line needs a patch under numpy >= 1.16: ``np.negative`` on a
    boolean mask (F7, step ellipsoid) becomes ``~``. The patch is made
    in memory, so the downloaded file stays as published.
    """
    source = path.read_text().replace("np.negative(idx)", "~idx")
    module = ModuleType("bbobbenchmarks")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def relative_deviation(values, reference, floor: float = 1.0) -> float:
    """Largest ``|values - reference| / max(|reference|, floor)``."""
    values = np.asarray(values)
    reference = np.asarray(reference)
    scale = np.maximum(np.abs(reference), floor)
    return float(np.max(np.abs(values - reference) / scale))
