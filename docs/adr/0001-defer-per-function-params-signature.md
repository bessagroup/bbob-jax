# ADR 0001: Keep the shared `fn(x, x_opt, f_opt, R, Q)` internal signature; defer `fn(x, params)`

- Status: accepted
- Date: 2026-07-09

## Context

The 49 function implementations nominally share the internal signature
`fn(x, x_opt, f_opt, R, Q)`, but the sharing is leaky:

- Two functions (`sharp_ridge`, `lunacek_bi_rastrigin`) order `f_opt` after
  `R, Q`; only keyword binding in the factories masks this.
- The four separable BBOB functions never read `R` or `Q`.
- Several functions reuse `Q[0, 0]` as an RNG seed via `jr.fold_in`
  (`linear_slope`, `schwefel_xsinx`, `lunacek_bi_rastrigin`,
  `_precompute_gallagher`) — a rotation matrix doubling as a seed carrier.
- Roughly ten optional precompute keyword families (`_mat`, `_gal_*`,
  `_sw_*`, `_f_max`, …) bolt onto the shared prefix, and the noisy CEC
  functions insert a positional `key`.

An architecture review (2026-07-09) proposed moving the seam to
`fn(x, params)` with a per-function params pytree declared in the
`FunctionSpec` table.

## Decision

Deferred. The uniform-prefix signature stays, for two load-bearing reasons:

1. **The flat function exports are public interface.** All 24 BBOB functions
   are exported at the package root and documented in `docs/api.md` with the
   `fn(x, x_opt, f_opt, R, Q)` signature. Changing it is a breaking change to
   every direct caller, for a purely internal tidiness gain.
2. **Problem instances are pinned to keys downstream.** The L2CO databanks
   and task datasets identify problem instances by `(function name, ndim,
   key)`. Any change to how parameters are derived from the key — including
   replacing the `Q[0, 0]` fold-in seeding with an honest key argument —
   silently changes every stored instance. The derivation is effectively
   frozen by data.

## Consequences

- The `FunctionSpec.x_opt_from` resolvers and the factory keyword binding
  carry the per-function knowledge instead; callers use `problem()` and
  never see the internal signature.
- The `Q[0, 0]`-as-seed hack must not be "fixed" in isolation — it is part
  of the frozen derivation (see reason 2).
- Revisit only alongside a major version bump that regenerates downstream
  databanks.
- Amended in part by ADR 0005 (2026-07-15): the *values* (and, for F4 and
  the Gallagher functions, parts of the derivation) were corrected to match
  the official reference, as a deliberate breaking change. The signature
  decision above stands.
