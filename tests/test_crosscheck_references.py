"""The cross-checks skip without their reference, and fail in CI.

CI sets ``BBOB_JAX_REQUIRE_REFERENCES=1`` so a cross-check whose
reference failed to download fails the run instead of skipping quietly.
"""

import pytest

from tests._references import (
    ENV_DIR,
    ENV_REQUIRE,
    LEGACY_PY,
    reference_path,
)


def test_missing_reference_skips(monkeypatch):
    monkeypatch.delenv(ENV_DIR, raising=False)
    monkeypatch.delenv(ENV_REQUIRE, raising=False)
    with pytest.raises(pytest.skip.Exception, match=ENV_DIR):
        reference_path(LEGACY_PY)


def test_missing_reference_fails_when_required(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_REQUIRE, "1")
    with pytest.raises(pytest.fail.Exception, match="not found"):
        reference_path(LEGACY_PY)


def test_present_reference_is_returned(monkeypatch, tmp_path):
    (tmp_path / LEGACY_PY).write_text("")
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    assert reference_path(LEGACY_PY) == tmp_path / LEGACY_PY
