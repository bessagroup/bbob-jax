# ADR 0002: One `FunctionSpec` table is the single source of truth per function

- Status: accepted
- Date: 2026-07-09

## Context

The four registries were hand-written dict literals (98 `Partial` entries),
with six randomized/deterministic factory twins whose bodies differed by one
line, plus separate hand-typed tag dicts (`defaultdict(dict)`, silently
returning `{}` on typos) and bounds dicts. Adding one function touched 7–9
sites; nothing but string equality linked them, and two BBOB tags shipped
wrong (`rastrigin_seperable`, `skew_rastrigin_bueche` labeled unimodal).
The CEC 2005 implementation plan (docs/superpowers/plans/2026-03-17) had
prescribed mirroring the BBOB registry layout, which is how the parallel
dicts accumulated.

## Decision

`_src/spec.py` holds one `FunctionSpec` row per function: implementation,
mode-parameterized maker (`deterministic=` flag replaces the twin
factories), tags, bounds, and an `x_opt_from` resolver locating the true
optimum. The registries, tag dicts and bounds dicts are derived views; the
tag dicts are plain dicts that raise on unknown names. `problem()` exposes
the whole row as one lookup. The refactor is bit-exact: key-split order and
all sampled parameters are unchanged (verified against golden values from
the pre-refactor registries — 1,960 arrays, zero mismatches).

## Consequences

- Adding a function = implementing it + adding one spec row (tests
  parametrize off the registries and pick it up automatically).
- Key drift between registries/tags/bounds is structurally impossible;
  `tests/test_consistency.py` pins the invariant and the tag schemas.
- The randomized/deterministic pair are two adapters of one factory
  family; the deterministic twins were deleted.
- Downstream key-pinned problem instances are unaffected (see the golden
  equivalence note above and ADR 0001).
