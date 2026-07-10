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
    cec2017_bounds,
    cec2017_function_characteristics,
    cec2017_registry,
    cec2017_registry_original,
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
CEC2017_TAG_KEYS = {
    "unimodal",
    "multimodal",
    "hybrid",
    "composition",
    "rotated",
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


def test_cec2017_key_sets_are_identical():
    names = set(cec2017_registry.keys())
    assert set(cec2017_registry_original.keys()) == names
    assert set(cec2017_function_characteristics.keys()) == names
    assert set(cec2017_bounds.keys()) == names


def test_suites_do_not_overlap():
    assert set(registry.keys()).isdisjoint(cec2005_registry.keys())
    assert set(registry.keys()).isdisjoint(cec2017_registry.keys())
    assert set(cec2005_registry.keys()).isdisjoint(cec2017_registry.keys())


@pytest.mark.parametrize(
    "tags_dict",
    [
        function_characteristics,
        cec2005_function_characteristics,
        cec2017_function_characteristics,
    ],
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


def test_cec2017_tag_schema():
    for name, tags in cec2017_function_characteristics.items():
        assert set(tags.keys()) == CEC2017_TAG_KEYS, name
        assert tags["multimodal"] == (not tags["unimodal"]), name
        if tags["hybrid"] or tags["composition"]:
            assert tags["multimodal"], name
        assert not (tags["hybrid"] and tags["composition"]), name


def test_cec2017_unimodal_set():
    """Only F1 (Bent Cigar) and F3 (Zakharov) are unimodal."""
    unimodal = {
        name
        for name, tags in cec2017_function_characteristics.items()
        if tags["unimodal"]
    }
    assert unimodal == {"cec2017_f1", "cec2017_f3"}


def test_cec2017_f6_is_not_rotated():
    """Regression pin: the reference code never applies F6's rotation
    (the kernel reads the pre-rotation shift buffer), so the instance
    is shift-only."""
    assert not cec2017_function_characteristics["cec2017_f6"]["rotated"]
    rotated_rest = {
        name
        for name, tags in cec2017_function_characteristics.items()
        if tags["rotated"]
    }
    assert rotated_rest == set(cec2017_function_characteristics) - {
        "cec2017_f6"
    }
