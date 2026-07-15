# BBOB-noisy suite: fn_true on every Problem, statistical noise verification

Status: accepted

The BBOB-noisy suite (f101–f130, registry keys `bbob_noisy_f101`…
`bbob_noisy_f130`) exposes the noisy evaluation as `fn(x, key)` — the
established CEC 2005 noisy convention — and additionally exposes the
undisturbed value (Ftrue) as a new `Problem.fn_true` field. `fn_true`
is uniform across all suites: for noise-free functions it is `fn`
itself; the four CEC 2005 noisy functions (F4/F17/F24/F25) were
retrofitted with an undisturbed path. Rationale: COCO-style evaluation
measures targets on Ftrue while the optimizer sees Fval, and Ftrue is
unrecoverable from the noise-free suite because the noisy base
parametrizations differ (conditioning, ×100 boundary penalty on every
function).

## Considered options

- **Optional key** (`fn(x)` → Ftrue, `fn(x, key)` → Fval): rejected —
  breaks the CEC 2005 required-key convention, and a forgotten key
  silently benchmarks against the noiseless function.
- **Not exposing Ftrue**: rejected — true-progress metrics (ERT-style)
  become impossible downstream.

## Verification

The deterministic Ftrue path is cross-checked against the compiled
legacy C reference (`benchmarksnoisy.c`) by binding reference-derived
instance parameters into the raw fns, as for CEC 2017 (ADR 0003):
`scripts/crosscheck_bbob_noisy.py`, all 30 functions at D in {2, 5, 10},
2 instances, worst relative deviation ~4e-12 (Griewank-Rosenbrock
accumulation-order noise; the official legacy Python deviates from the C
by the same amount). The stochastic Fval path cannot be matched bitwise
(the C reference uses its own RNG), so noise models are verified by
exact formula unit tests with pinned JAX draws plus statistical tests
against the definition. Code-wins on any definition-vs-reference
discrepancy.

Note: the *current* COCO revival (`transform_obj_*_noise.c`) deviates
from the legacy code — it drops the noise gate and uses a linear
(unsquared) boundary penalty. The legacy code agrees with the published
definition on both points, so legacy semantics are kept. The suite
originally carried local `_tosz`/`_asym` transforms because the shared
`transforms.py` variants deviated from the reference; since ADR 0005
corrected the shared transforms, the suite reuses them (the C
cross-check is unchanged).
