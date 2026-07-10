import math

import pytest

from bbob_jax import bbob_bounds, cec2005_bounds, cec2017_bounds, registry
from bbob_jax._src.bounds import BBOB_BOUNDS


def test_bbob_bounds_keys():
    """bbob_bounds has exactly the same keys as registry."""
    assert set(bbob_bounds.keys()) == set(registry.keys())


def test_bbob_bounds_values():
    """All BBOB bounds are (-5.0, 5.0)."""
    for name, bounds in bbob_bounds.items():
        assert bounds == (-5.0, 5.0), (
            f"{name}: expected (-5.0, 5.0), got {bounds}"
        )


def test_cec2005_bounds_keys():
    """cec2005_bounds has entries for all 25 CEC 2005 functions f1–f25."""
    assert set(cec2005_bounds.keys()) == {f"f{i}" for i in range(1, 26)}


def test_bounds_count():
    """Correct number of entries."""
    assert len(bbob_bounds) == 24
    assert len(cec2005_bounds) == 25


def test_no_key_overlap():
    """BBOB and CEC 2005 function names do not overlap."""
    assert set(bbob_bounds.keys()).isdisjoint(set(cec2005_bounds.keys()))


def test_cec2017_bounds_keys():
    """cec2017_bounds covers the full suite: F1 and F3-F30 (F2 was
    officially withdrawn)."""
    assert set(cec2017_bounds.keys()) == {
        f"cec2017_f{i}" for i in range(1, 31) if i != 2
    }


def test_cec2017_bounds_values():
    """All CEC 2017 bounds are (-100.0, 100.0)."""
    for name, bounds in cec2017_bounds.items():
        assert bounds == (-100.0, 100.0), (
            f"{name}: expected (-100.0, 100.0), got {bounds}"
        )


def test_constants():
    assert BBOB_BOUNDS == (-5.0, 5.0)


@pytest.mark.parametrize("name", list(registry.keys()))
def test_bbob_lookup_by_name(name):
    assert bbob_bounds[name] == (-5.0, 5.0)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("f1", (-100.0, 100.0)),
        ("f7", (0.0, 600.0)),
        ("f8", (-32.0, 32.0)),
        ("f9", (-5.0, 5.0)),
        ("f11", (-0.5, 0.5)),
        ("f12", (-math.pi, math.pi)),
        ("f13", (-3.0, 1.0)),
        ("f25", (2.0, 5.0)),
    ],
)
def test_cec2005_lookup_by_name(name, expected):
    assert cec2005_bounds[name] == expected
