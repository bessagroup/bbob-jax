# CEC 2013 LSGO reference constants — provenance

The `F{1..15}.npz` files in this directory are the **official constants**
that define the CEC 2013 Large-Scale Global Optimization (LSGO) benchmark:
per-function shift vectors (`xopt`), subcomponent rotation matrices
(`R25`/`R50`/`R100`), variable permutations (`p`, stored 1-indexed as in the
reference), subcomponent sizes (`s`) and weights (`w`).

Unlike `bbob-jax`'s CEC 2005 / CEC 2017 suites — whose instance parameters
are generated from JAX PRNG keys — these constants **cannot be regenerated
from a seed**: they *are* the benchmark, and any optimizer comparison that
claims to be "CEC 2013 LSGO" must use exactly these values. They are
therefore vendored as package data so the suite is self-contained and
reproducible off-cluster.

## Chain of provenance

| Layer | Source |
|---|---|
| Original benchmark | Li, Tang, Omidvar, Yang & Qin, *Benchmark Functions for the CEC 2013 Special Session and Competition on Large-Scale Global Optimization*, Technical Report, 2013. |
| Reference implementation & data | Daniel Molina, `cec2013lsgo` — <https://github.com/dmolina/cec2013lsgo> (a Python wrapper over the official C/MATLAB code; the `.txt` constants originate here / from the official competition materials). |
| Immediate source of the copy | MetaBox — `MetaEvo/MetaBox@5565a28` (v2.0.1+5), path `src/environment/problem/SOO/CEC2013LSGO/datafile/`. BSD 3-Clause, © 2023 MetaEvolution Lab. |

## How these files were produced

`scripts/convert_cec2013lsgo_data.py` read MetaBox's 75 plain-text datafiles
and wrote one compressed `.npz` per function (2.9 MB of text → ~1.1 MB).
Values are stored as `float64` (`xopt`, `R*`, `w`) and `int32` (`p`, `s`);
the permutation is kept verbatim (1-indexed) and de-indexed in
`cec2013lsgo.py`. F13 is 905-D; F14 keeps its 1000-value `xopt` (split into
per-subcomponent optima at load time).

## Licensing note

These arrays are numerical constants of a public academic benchmark, not
source code; they are redistributed here for research reproducibility with
attribution to all three layers above. The benchmark constants carry the
terms of the upstream competition materials / the `cec2013lsgo` reference
(consult that repository for its exact license); MetaBox's own BSD 3-Clause
license and copyright are reproduced in the repository-root
`THIRD_PARTY_NOTICES.md`. If any upstream terms conflict with
redistribution, this directory — not the rest of `bbob-jax` — is the part to
revisit.
