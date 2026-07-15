# bbob-jax

JAX implementations of black-box optimization benchmark suites (BBOB
noise-free, BBOB-noisy, CEC 2005, CEC 2017), exposing differentiable,
JIT-able, vmap-able problem instances.

## Language

**Suite**:
A published benchmark family (bbob, bbob_noisy, cec2005, cec2017). Each
suite has its own registry pair, tag schema and bounds dict, all derived
from the spec table.

**Spec row**:
The single `FunctionSpec` entry that defines one benchmark function —
implementation, maker, tags, bounds, optimum resolver. Adding a function
means adding one spec row.

**Instance**:
A concrete problem constructed by a maker from `(ndim, key)` — the
function with sampled `x_opt`, `f_opt` and rotations bound in.
_Avoid_: COCO's integer "instance ID" — bbob-jax instances are keyed by
JAX PRNG keys, not by instance numbers.

**Deterministic instance**:
The instance built with `deterministic=True`: zero shift, identity
rotations, zero `f_opt`. For noisy functions the *parameters* are
deterministic but evaluation stays stochastic.
_Avoid_: "noiseless instance" — determinism refers to parameters, never
to noise.

**Noisy function**:
A function whose evaluation consumes a PRNG key (`fn(x, key)`), marked
by the `noise` tag. Covers BBOB-noisy f101–f130 and CEC 2005
F4/F17/F24/F25.

**Undisturbed value (Ftrue)**:
The noise-free function value — base + boundary penalty + `f_opt` — of a
noisy function. Exposed as `Problem.fn_true`; for noise-free functions
`fn_true` is `fn` itself. Used by harnesses to measure true progress.
_Avoid_: "noiseless value", "clean value"

**Disturbed value (Fval)**:
What `fn(x, key)` returns for a noisy function: the noise model applied
to the undisturbed residual, plus penalty and `f_opt`. This is all an
optimizer is allowed to see.

**Noise model**:
One of the three BBOB-noisy stochastic transforms of the undisturbed
residual: Gaussian (multiplicative), uniform (multiplicative), Cauchy
(additive).

**Severity**:
The BBOB-noisy parameter regime of a noise model: *moderate*
(f101–f106) or *severe* (f107–f130). Tagged as the boolean `severe`.

**Noise gate**:
The BBOB-noisy final adjustment: if the undisturbed residual is below
1e-8 the undisturbed value is returned untouched; otherwise the
disturbed value plus 1.01e-8. Guarantees `fn(x_opt, key) == f_opt`
exactly.
_Avoid_: "final adjustment" (the def-page term; gate is what it does)

**Boundary penalty**:
The out-of-bounds quadratic penalty term. BBOB-noisy applies it with
factor 100 to every function; it is added outside the noise model, so
it is never noisy.

**Code-wins rule**:
When the published definition and the official reference implementation
disagree, replicate the reference code, quirks included.
