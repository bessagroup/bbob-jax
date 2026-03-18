import pytest
from bbob_jax import bbob_bounds, cec2005_bounds
from bbob_jax._src.bounds import BBOB_BOUNDS, CEC2005_BOUNDS
from bbob_jax import registry, cec2005_registry


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
    """cec2005_bounds has exactly the same keys as cec2005_registry."""
    assert set(cec2005_bounds.keys()) == set(cec2005_registry.keys())


def test_cec2005_bounds_values():
    """All CEC 2005 bounds are (-100.0, 100.0)."""
    for name, bounds in cec2005_bounds.items():
        assert bounds == (-100.0, 100.0), (
            f"{name}: expected (-100.0, 100.0), got {bounds}"
        )


def test_bounds_count():
    """Correct number of entries."""
    assert len(bbob_bounds) == 24
    assert len(cec2005_bounds) == 25


def test_no_key_overlap():
    """BBOB and CEC 2005 function names do not overlap."""
    assert set(bbob_bounds.keys()).isdisjoint(set(cec2005_bounds.keys()))


def test_constants():
    assert BBOB_BOUNDS == (-5.0, 5.0)
    assert CEC2005_BOUNDS == (-100.0, 100.0)


@pytest.mark.parametrize("name", list(registry.keys()))
def test_bbob_lookup_by_name(name):
    assert bbob_bounds[name] == (-5.0, 5.0)


@pytest.mark.parametrize("name", [f"f{i}" for i in range(1, 26)])
def test_cec2005_lookup_by_name(name):
    assert cec2005_bounds[name] == (-100.0, 100.0)
