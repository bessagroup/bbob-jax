# Reference-faithful noiseless BBOB suite (breaking value correction)

- Status: accepted
- Date: 2026-07-15

## Context

A systematic audit against the official BBOB reference implementation
(legacy `bbobbenchmarks.py`, itself regression-pinned to the legacy C;
see `scripts/crosscheck_bbob_noiseless.py`) found that 11 of the 24
noiseless functions deviated from the reference, all dating to the
initial implementation:

- `tosz_func` applied T_osz only to components equal to the first/last
  element — a misparse of the definition's "for any positive integer n
  (n = 1 and n = D are used in the following)" — and used the
  positive-branch constants (10, 7.9) for negative inputs instead of
  (5.5, 3.1). Affected F2, F10, F11, F16 (22%–48% relative deviation).
- `tasy_func` used exponent `(i-1)/(D-1)` with 0-based `i` — an
  off-by-one from the paper's 1-based notation (the reference uses
  `beta * linspace(0, 1, D)`). Affected F12, F17, F18 (79%–790%).
- F3/F15 multiplied the Rastrigin cosine term by `sum(z^2)` instead of
  adding it. Beyond unfaithfulness this made **every integer lattice
  point of z a global minimum with value exactly `f_opt`** (both
  factors vanish), corrupting optimum-hitting metrics.
- F4 skewed the wrong coordinate parity (0-based odd instead of even,
  plus a silent out-of-bounds index clip) and did not make the skewed
  optimum coordinates non-negative.
- F14 omitted the final square root.
- Gallagher's per-peak conditioning diagonals were not permuted.

## Decision

Fix all of it, in place, as a deliberate breaking change: function
values change for the affected functions under identical keys, and
instance parameters change for F4 (optimum evening) and the Gallagher
functions (permuted conditioning). This amends ADR 0001's freeze —
which anticipated exactly this: *"revisit only alongside a major
version bump that regenerates downstream databanks."* Downstream
databanks, trained policies and stored trajectories that used the
affected functions must be regenerated; values from older versions are
not comparable.

The alternative — a coexisting "faithful" suite next to the warty one —
was rejected: it would permanently double the noiseless surface to
preserve landscapes that are defective (F3/F15 degenerate minima), not
merely different.

## Verification

`scripts/crosscheck_bbob_noiseless.py` binds reference-derived instance
parameters into all 24 implementations and compares point-for-point:
worst relative deviation 1.5e-10 (F19, accumulation order), everything
else ≤ 5.4e-13. The BBOB-noisy suite now reuses the corrected
`tosz_func`/`tasy_func` (its own compiled-C cross-check is unchanged at
~4.4e-12). `tests/test_reference_fidelity.py` pins the corrections in
CI without the external reference.

Remaining deliberate deviations from the reference (unchanged):
F9/F19/F20/F24 add an optional instance shift the reference lacks
(shift 0 reproduces the reference exactly), and instance parameters are
sampled from JAX keys rather than derived from COCO instance IDs.
