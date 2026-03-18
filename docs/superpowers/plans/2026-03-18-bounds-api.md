# Bounds API Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose box-constraint bounds (search domain boundaries) for all BBOB and CEC 2005 benchmark functions as a public API, and use them as defaults in the plotting functions.

**Architecture:** A new `bounds.py` module in `_src/` defines the two bound constants and builds per-function dicts by iterating the existing registry keys; the dicts are re-exported from the top-level `__init__.py`. The plotting module imports `BBOB_BOUNDS` from that module and uses it as the runtime default when `bounds=None` is passed.

**Tech Stack:** Python 3.10+, JAX, pytest, Ruff (lint/format).

---

## File Map

| File | Change |
|---|---|
| `src/bbob_jax/_src/bounds.py` | CREATE — constants + bounds dicts |
| `src/bbob_jax/__init__.py` | EDIT — import + `__all__` entries |
| `src/bbob_jax/_src/plotting.py` | EDIT — `bounds=None` default + runtime fallback |
| `tests/test_bounds.py` | CREATE — all bounds tests |

---

## Chunk 1: Bounds dicts and public API

### Task 1: Create `bounds.py`, write and pass tests, export from `__init__.py`

- [ ] **1.1 — Write the test file first (TDD red phase)**

  Create `tests/test_bounds.py` with the following content:

  ```python
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
          assert bounds == (-5.0, 5.0), f"{name}: expected (-5.0, 5.0), got {bounds}"


  def test_cec2005_bounds_keys():
      """cec2005_bounds has exactly the same keys as cec2005_registry."""
      assert set(cec2005_bounds.keys()) == set(cec2005_registry.keys())


  def test_cec2005_bounds_values():
      """All CEC 2005 bounds are (-100.0, 100.0)."""
      for name, bounds in cec2005_bounds.items():
          assert bounds == (-100.0, 100.0), f"{name}: expected (-100.0, 100.0), got {bounds}"


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
  ```

- [ ] **1.2 — Confirm tests fail (expected: ImportError or AttributeError)**

  ```bash
  uv run --no-sync pytest tests/test_bounds.py -v 2>&1 | head -20
  ```

  Expected: collection errors because `bbob_bounds` / `cec2005_bounds` do not exist yet.

- [ ] **1.3 — Create `src/bbob_jax/_src/bounds.py`**

  ```python
  #                                                                       Modules
  # =============================================================================

  # Local
  from bbob_jax._src.registry import cec2005_registry, registry

  #                                                          Authorship & Credits
  # =============================================================================
  __author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
  __credits__ = ["Martin van der Schelling"]
  __status__ = "Stable"
  # =============================================================================

  BBOB_BOUNDS: tuple[float, float] = (-5.0, 5.0)
  CEC2005_BOUNDS: tuple[float, float] = (-100.0, 100.0)

  bbob_bounds: dict[str, tuple[float, float]] = {
      name: BBOB_BOUNDS for name in registry
  }

  cec2005_bounds: dict[str, tuple[float, float]] = {
      name: CEC2005_BOUNDS for name in cec2005_registry
  }
  ```

- [ ] **1.4 — Export from `src/bbob_jax/__init__.py`**

  Add the import line (alongside other `_src` imports):

  ```python
  from bbob_jax._src.bounds import bbob_bounds, cec2005_bounds
  ```

  Add to `__all__`:

  ```python
  "bbob_bounds",
  "cec2005_bounds",
  ```

- [ ] **1.5 — Run bounds tests (TDD green phase)**

  ```bash
  uv run --no-sync pytest tests/test_bounds.py -v
  ```

  Expected: all tests pass (7 non-parametrized + 24 parametrized BBOB + 25 parametrized CEC 2005 = 56 tests).

- [ ] **1.6 — Run full test suite to confirm no regressions**

  ```bash
  uv run --no-sync pytest --tb=short -q 2>&1 | tail -10
  ```

  Expected: all previously passing tests still pass (1612 + 56 new).

- [ ] **1.7 — Commit**

  ```bash
  git add src/bbob_jax/_src/bounds.py src/bbob_jax/__init__.py tests/test_bounds.py
  git commit -m "feat: add bbob_bounds and cec2005_bounds public API"
  ```

---

## Chunk 2: Plotting defaults and final validation

### Task 2: Update `plot_2d` and `plot_3d` to use `bounds=None` with BBOB_BOUNDS fallback

- [ ] **2.1 — Add import to `src/bbob_jax/_src/plotting.py`**

  After the existing `from .utils import _create_mesh` import line, add:

  ```python
  from .bounds import BBOB_BOUNDS
  ```

- [ ] **2.2 — Update `plot_2d` signature**

  Change:

  ```python
      bounds: tuple[float, float] = (-5.0, 5.0),
  ```

  to:

  ```python
      bounds: tuple[float, float] | None = None,
  ```

- [ ] **2.3 — Add `bounds` fallback at top of `plot_2d` body**

  Insert before the `fn_instance, _ = fn(ndim=2, key=key)` line:

  ```python
      if bounds is None:
          bounds = BBOB_BOUNDS
  ```

- [ ] **2.4 — Update `plot_2d` docstring**

  Change:

  ```
      bounds : tuple[float, float], optional
          Min and max values for both x and y axes, by default (-5.0, 5.0).
  ```

  to:

  ```
      bounds : tuple[float, float] | None, optional
          Min and max values for both x and y axes. Defaults to BBOB_BOUNDS
          (-5.0, 5.0) when None. Pass ``cec2005_bounds[name]`` for CEC 2005
          functions.
  ```

- [ ] **2.5 — Apply the same three changes to `plot_3d`**

  - Change signature: `bounds: tuple[float, float] | None = None`
  - Add `if bounds is None: bounds = BBOB_BOUNDS` before first use of `bounds`
  - Update docstring as above

- [ ] **2.6 — Run full test suite**

  ```bash
  uv run --no-sync pytest --tb=short -q 2>&1 | tail -10
  ```

  Expected: all tests pass with no regressions.

- [ ] **2.7 — Lint and format**

  ```bash
  make lint
  make format
  ```

  Expected: no lint errors; format may auto-fix whitespace. If format changes files, re-stage them.

- [ ] **2.8 — Commit**

  ```bash
  git add src/bbob_jax/_src/plotting.py
  git commit -m "feat: use BBOB_BOUNDS as default in plot_2d and plot_3d"
  ```

---

## Chunk review notes

Both chunks were reviewed against the spec:

**Chunk 1** — APPROVED. TDD order is correct (tests written before implementation). All 56 expected tests are covered. File paths match the spec. Each registry key is covered via parametrize. The `bounds.py` module correctly derives dicts from live registry keys, so additions to the registry are automatically reflected.

**Chunk 2** — APPROVED. The plotting change is backward-compatible (existing callers passing explicit bounds are unaffected). The `BBOB_BOUNDS` fallback is applied at runtime rather than as a default argument value, which avoids the mutable-default-argument anti-pattern and allows the constant to be the single source of truth. Docstring update guides users toward the correct pattern for CEC 2005 functions.
