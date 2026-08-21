# Third-party notices

`bbob-jax` incorporates work derived from third-party projects. Their
licenses and copyright notices are reproduced below, as required.

---

## MetaBox

The CEC 2013 Large-Scale Global Optimization suite (`_src/cec2013lsgo.py`
and the reference constants in `_src/cec2013lsgo_data/`) was **ported to
JAX from MetaBox's NumPy implementation**.

- Project: MetaBox — <https://github.com/MetaEvo/MetaBox>
- Version: `MetaEvo/MetaBox@5565a28` (v2.0.1+5)
- Source ported: `src/environment/problem/SOO/CEC2013LSGO/cec2013lsgo_numpy.py`
- Data copied:   `src/environment/problem/SOO/CEC2013LSGO/datafile/`
- License: BSD 3-Clause, © 2023 MetaEvolution Lab

If you use this suite, please cite MetaBox (and the original benchmark and
its reference implementation — see below and the README):

- Z. Ma et al., "MetaBox: A Benchmark Platform for Meta-Black-Box
  Optimization with Reinforcement Learning", NeurIPS 2023.
  arXiv:2311.02708.
- MetaBox-v2, NeurIPS 2025. arXiv:2505.17745.

```
BSD 3-Clause License

Copyright (c) 2023, MetaEvolution Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Upstream reference implementation and data

MetaBox's CEC 2013 LSGO code and datafiles derive from Daniel Molina's
`cec2013lsgo` reference implementation
(<https://github.com/dmolina/cec2013lsgo>), which wraps the official
CEC 2013 competition C/MATLAB code. The vendored numerical constants
(`_src/cec2013lsgo_data/*.npz`) originate there; consult that repository
for the terms governing the benchmark data, and see
`_src/cec2013lsgo_data/PROVENANCE.md` for the full chain of provenance.

---

## Provenance table

| bbob-jax component | Derived from | Original benchmark |
|---|---|---|
| `_src/cec2013lsgo.py` | MetaBox `cec2013lsgo_numpy.py` @ 5565a28 (BSD-3) | Li, Tang, Omidvar, Yang & Qin, CEC 2013 LSGO (2013) |
| `_src/cec2013lsgo_data/*.npz` | MetaBox `CEC2013LSGO/datafile/` @ 5565a28 → dmolina `cec2013lsgo` | official CEC 2013 competition data |
