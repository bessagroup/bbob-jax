# CEC 2005 Per-Function Bounds Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uniform CEC 2005 `(-100, 100)` bounds with per-function bounds from the benchmark paper, and revert the plotting default back to a hardcoded `(-5.0, 5.0)`.

**Architecture:** `bounds.py` drops `CEC2005_BOUNDS` and replaces the registry-derived comprehension with an explicit dict keyed `f1`–`f25`. The plotting module removes its `BBOB_BOUNDS` import and reverts both `plot_2d` / `plot_3d` to the original hardcoded `(-5.0, 5.0)` default. Tests in `test_bounds.py` are updated in-place to match the new per-function values.

**Tech Stack:** Python 3.10+, JAX, pytest, Ruff (lint/format).

---

## File Map

| File | Change |
|---|---|
| `src/bbob_jax/_src/bounds.py` | EDIT — remove `CEC2005_BOUNDS`, remove `cec2005_registry` import, replace `cec2005_bounds` dict comprehension with explicit per-function dict |
| `src/bbob_jax/_src/plotting.py` | EDIT — remove `BBOB_BOUNDS` import, revert both function signatures and bodies to hardcoded `(-5.0, 5.0)`, revert docstrings |
| `tests/test_bounds.py` | EDIT — remove `CEC2005_BOUNDS` import, update `test_cec2005_bounds_values` and `test_constants` and `test_cec2005_lookup_by_name` |

---

## Chunk 1: Per-function CEC 2005 bounds

### Task 1: Update tests (TDD red), then update `bounds.py` to pass them

- [ ] **1.1 — Update `tests/test_bounds.py` (TDD red phase)**

  Make these four targeted edits:

  **Edit A — replace the two import lines (lines 3–4)** from:
  ```python
  from bbob_jax._src.bounds import BBOB_BOUNDS, CEC2005_BOUNDS
  from bbob_jax import registry, cec2005_registry
  ```
  to:
  ```python
  import math

  from bbob_jax._src.bounds import BBOB_BOUNDS
  from bbob_jax import registry
  ```

  **Edit B — replace `test_cec2005_bounds_keys`** (lines 20–22) with:
  ```python
  def test_cec2005_bounds_keys():
      """cec2005_bounds has entries for all 25 CEC 2005 functions f1–f25."""
      assert set(cec2005_bounds.keys()) == {f"f{i}" for i in range(1, 26)}
  ```

  **Edit C — replace `test_cec2005_bounds_values`** (lines 25–30) with:
  ```python
  def test_cec2005_bounds_values():
      """CEC 2005 bounds are per-function as specified in the benchmark paper."""
      expected = {
          "f1": (-100.0, 100.0),
          "f2": (-100.0, 100.0),
          "f3": (-100.0, 100.0),
          "f4": (-100.0, 100.0),
          "f5": (-100.0, 100.0),
          "f6": (-100.0, 100.0),
          "f7": (0.0, 600.0),
          "f8": (-32.0, 32.0),
          "f9": (-5.0, 5.0),
          "f10": (-5.0, 5.0),
          "f11": (-0.5, 0.5),
          "f12": (-math.pi, math.pi),
          "f13": (-3.0, 1.0),
          "f14": (-100.0, 100.0),
          "f15": (-5.0, 5.0),
          "f16": (-5.0, 5.0),
          "f17": (-5.0, 5.0),
          "f18": (-5.0, 5.0),
          "f19": (-5.0, 5.0),
          "f20": (-5.0, 5.0),
          "f21": (-5.0, 5.0),
          "f22": (-5.0, 5.0),
          "f23": (-5.0, 5.0),
          "f24": (-5.0, 5.0),
          "f25": (2.0, 5.0),
      }
      assert cec2005_bounds == expected
  ```

  **Edit D — replace `test_constants`** (lines 44–46) with:
  ```python
  def test_constants():
      assert BBOB_BOUNDS == (-5.0, 5.0)
  ```

  **Edit E — replace `test_cec2005_lookup_by_name`** (lines 54–56) with:
  ```python
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
  ```

- [ ] **1.2 — Confirm updated tests fail (expected)**

  ```bash
  uv run --no-sync pytest tests/test_bounds.py -v 2>&1 | head -30
  ```

  Expected: `test_cec2005_bounds_keys`, `test_cec2005_bounds_values`, and `test_cec2005_lookup_by_name` fail because `cec2005_bounds` still returns `(-100.0, 100.0)` for all keys; `test_constants` fails with `ImportError` on `CEC2005_BOUNDS`.

- [ ] **1.3 — Rewrite `src/bbob_jax/_src/bounds.py`**

  Replace the entire file content with:

  ```python
  #                                                                       Modules
  # =============================================================================

  # Standard
  import math

  # Local
  from bbob_jax._src.registry import registry

  #                                                          Authorship & Credits
  # =============================================================================
  __author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
  __credits__ = ["Martin van der Schelling"]
  __status__ = "Stable"
  # =============================================================================

  BBOB_BOUNDS: tuple[float, float] = (-5.0, 5.0)

  bbob_bounds: dict[str, tuple[float, float]] = {
      name: BBOB_BOUNDS for name in registry
  }

  cec2005_bounds: dict[str, tuple[float, float]] = {
      "f1": (-100.0, 100.0),
      "f2": (-100.0, 100.0),
      "f3": (-100.0, 100.0),
      "f4": (-100.0, 100.0),
      "f5": (-100.0, 100.0),
      "f6": (-100.0, 100.0),
      "f7": (0.0, 600.0),
      "f8": (-32.0, 32.0),
      "f9": (-5.0, 5.0),
      "f10": (-5.0, 5.0),
      "f11": (-0.5, 0.5),
      "f12": (-math.pi, math.pi),
      "f13": (-3.0, 1.0),
      "f14": (-100.0, 100.0),
      "f15": (-5.0, 5.0),
      "f16": (-5.0, 5.0),
      "f17": (-5.0, 5.0),
      "f18": (-5.0, 5.0),
      "f19": (-5.0, 5.0),
      "f20": (-5.0, 5.0),
      "f21": (-5.0, 5.0),
      "f22": (-5.0, 5.0),
      "f23": (-5.0, 5.0),
      "f24": (-5.0, 5.0),
      "f25": (2.0, 5.0),
  }
  ```

- [ ] **1.4 — Run bounds tests (TDD green phase)**

  ```bash
  uv run --no-sync pytest tests/test_bounds.py -v
  ```

  Expected: all tests pass. Count: 7 non-parametrized + 24 parametrized BBOB + 8 parametrized CEC 2005 spot-checks = 39 tests (down from 56: removed 25-entry CEC parametrize and 1 `CEC2005_BOUNDS` constant check, added 8 spot-checks and 1 updated `test_cec2005_bounds_values`).

- [ ] **1.5 — Run full test suite to confirm no regressions**

  ```bash
  uv run --no-sync pytest --tb=short -q 2>&1 | tail -5
  ```

  Expected: all tests pass. Total count drops from 1668 to 1651 (removed 25 old CEC parametrize entries + 1 `CEC2005_BOUNDS` constant check = −26; added 1 updated `test_cec2005_bounds_values` + 8 spot-checks = +9; net −17).

- [ ] **1.6 — Commit**

  ```bash
  git add src/bbob_jax/_src/bounds.py tests/test_bounds.py
  git commit -m "feat: use per-function bounds for CEC 2005 instead of uniform (-100, 100)"
  ```

---

## Chunk 2: Revert plotting defaults

### Task 2: Revert `plot_2d` and `plot_3d` to hardcoded `(-5.0, 5.0)` default

- [ ] **2.1 — Remove `BBOB_BOUNDS` import from `src/bbob_jax/_src/plotting.py`**

  Remove line 16:
  ```python
  from .bounds import BBOB_BOUNDS
  ```

- [ ] **2.2 — Revert `plot_2d` signature**

  Change (line 30):
  ```python
      bounds: tuple[float, float] | None = None,
  ```
  to:
  ```python
      bounds: tuple[float, float] = (-5.0, 5.0),
  ```

- [ ] **2.3 — Remove `bounds` fallback from `plot_2d` body**

  Remove lines 63–64:
  ```python
      if bounds is None:
          bounds = BBOB_BOUNDS
  ```

- [ ] **2.4 — Revert `plot_2d` docstring**

  Change (lines 45–48):
  ```
      bounds : tuple[float, float] | None, optional
          Min and max values for both x and y axes. Defaults to BBOB_BOUNDS
          (-5.0, 5.0) when None. Pass ``cec2005_bounds[name]`` for CEC 2005
          functions.
  ```
  to:
  ```
      bounds : tuple[float, float], optional
          Min and max values for both x and y axes, by default (-5.0, 5.0).
  ```

- [ ] **2.5 — Apply the same three changes to `plot_3d`**

  - Revert signature (line 97): `bounds: tuple[float, float] = (-5.0, 5.0)`
  - Remove `if bounds is None: bounds = BBOB_BOUNDS` (lines 127–128)
  - Revert docstring (lines 112–115) to: `bounds : tuple[float, float], optional\n        Min and max values for both x and y axes, by default (-5.0, 5.0).`

- [ ] **2.6 — Run full test suite**

  ```bash
  uv run --no-sync pytest --tb=short -q 2>&1 | tail -5
  ```

  Expected: all tests pass, same count as after Task 1.

- [ ] **2.7 — Lint and format**

  ```bash
  make lint
  make format
  ```

  Expected: no errors. If format auto-fixes whitespace, re-stage the file.

- [ ] **2.8 — Commit**

  ```bash
  git add src/bbob_jax/_src/plotting.py
  git commit -m "revert: restore hardcoded (-5.0, 5.0) default in plot_2d and plot_3d"
  ```

---

## Chunk review notes

**Chunk 1** — APPROVED. All 25 per-function CEC 2005 bound values match the spec in both the test edits and the `bounds.py` replacement. The `cec2005_registry` import is correctly removed from both `bounds.py` and `test_bounds.py`. `test_cec2005_bounds_keys` is updated to use a hardcoded set `{f"f{i}" for i in range(1, 26)}` so it no longer depends on the registry.

**Chunk 2** — APPROVED. All three reversions are covered: import removal, signature revert (both functions), fallback block removal (both functions), docstring revert (both functions). Line numbers verified against the current file state.
