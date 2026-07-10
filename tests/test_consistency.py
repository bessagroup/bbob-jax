"""Consistency gate: the derived metadata views stay in sync.

The registries, tag dicts and bounds dicts are all views of
the FunctionSpec table, so key drift is structurally
impossible — these tests pin that invariant (and the tag
schemas) against future hand edits.
"""

import pytest

from bbob_jax import (
    bbob_bounds,
    cec2005_bounds,
    cec2005_function_characteristics,
    cec2005_registry,
    cec2005_registry_original,
    function_characteristics,
    registry,
    registry_original,
)

BBOB_TAG_KEYS = {"separable", "unimodal"}
CEC_TAG_KEYS = {
    "unimodal",
    "multimodal",
    "composition",
    "rotated",
    "noise",
    "structure_modified",
}


def test_bbob_key_sets_are_identical():
    names = set(registry.keys())
    assert set(registry_original.keys()) == names
    assert set(function_characteristics.keys()) == names
    assert set(bbob_bounds.keys()) == names


def test_cec2005_key_sets_are_identical():
    names = set(cec2005_registry.keys())
    assert set(cec2005_registry_original.keys()) == names
    assert set(cec2005_function_characteristics.keys()) == names
    assert set(cec2005_bounds.keys()) == names


def test_suites_do_not_overlap():
    assert set(registry.keys()).isdisjoint(cec2005_registry.keys())


@pytest.mark.parametrize(
    "tags_dict", [function_characteristics, cec2005_function_characteristics]
)
def test_unknown_name_raises_key_error(tags_dict):
    """Plain dicts: a typo raises instead of returning {}."""
    with pytest.raises(KeyError):
        tags_dict["not_a_function"]


def test_bbob_tag_schema():
    for name, tags in function_characteristics.items():
        assert set(tags.keys()) == BBOB_TAG_KEYS, name
        assert all(isinstance(v, bool) for v in tags.values()), name


def test_cec2005_tag_schema():
    for name, tags in cec2005_function_characteristics.items():
        assert set(tags.keys()) == CEC_TAG_KEYS, name
        assert tags["multimodal"] == (not tags["unimodal"]), name
        if tags["composition"]:
            assert tags["multimodal"], name


def test_rastrigin_variants_are_multimodal():
    """Regression pin: these were mislabeled unimodal before the spec."""
    assert not function_characteristics["rastrigin_seperable"]["unimodal"]
    assert not function_characteristics["skew_rastrigin_bueche"]["unimodal"]
