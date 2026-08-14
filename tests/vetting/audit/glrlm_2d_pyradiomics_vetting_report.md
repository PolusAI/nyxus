# Audit: 2D GLRLM vs a fresh PyRadiomics run

**Verdict: all 16 quantities reproduce.** 15 to double precision (worst 2.3e-16), run entropy to
1.1e-3 - explained below and asserted at a band that matches.

Covers `tests/test_2d_glrlm_pyradiomics.h` (goldens + assertions), `tests/test_2d_glrlm_common.h`
(fixture) and `tests/vetting/oracles/gen_glrlm_pyradiomics.py` (generator).

## Method

- **Tool**: pyradiomics **v3.0.1** with SimpleITK, in a conda env (`conda create -n nyxus_oracle -c
  conda-forge python=3.9 pyradiomics simpleitk numpy`). Python 3.9 is required - conda-forge does
  not resolve pyradiomics on 3.11.
- **Config**: `binWidth=1` (identity binning on this integer phantom), `distances=[1]`,
  `force2D=True`, `force2Ddimension=0`, `weightingNorm=None`, `label=1`. Nyxus side: recipe
  `glrlm.ibsi_ng128` (`ibsi=true`, `GREYDEPTH=128`).
- **Fixture**: the IBSI digital phantom exactly as `tests/test_data.h` stores it, read out of that
  header by `tests/vetting/oracles/ibsi_phantom.py` - so the generator, the C++ test and the tool
  are fed one copy of the pixels and cannot drift apart. 4 slices, 20/19/17/18 masked voxels,
  in-mask grey levels {1,3,4,6}.
- **Aggregation**: PyRadiomics reports one value per feature over the 4 in-slice directions, which
  is the Nyxus `*_AVE` aggregation; averaging that over the 4 slices is the IBSI 2D-averaged
  aggregation. The per-direction features are checked as the mean of their 4 directional values.
- **Command**: `python tests/vetting/oracles/gen_glrlm_pyradiomics.py`; full steps in
  `glrlm_2d_golden_regen.md`.

## Result table

| feature | Nyxus | fresh run | rel | verdict |
|---|---|---|---:|---|
| `GLRLM_GLN` | 5.19706240451 | 5.19706240451 | 1.7e-16 | vetted |
| `GLRLM_GLNN` | 0.459729336675 | 0.459729336675 | 2.4e-16 | vetted |
| `GLRLM_GLV` | 3.35302839111 | 3.35302839111 | 0.0e+00 | vetted |
| `GLRLM_HGLRE` | 9.8242744541 | 9.8242744541 | 0.0e+00 | vetted |
| `GLRLM_LGLRE` | 0.604357907739 | 0.604357907739 | 0.0e+00 | vetted |
| `GLRLM_LRE` | 3.77838422301 | 3.77838422301 | 1.2e-16 | vetted |
| `GLRLM_LRHGLE` | 17.387027112 | 17.387027112 | 0.0e+00 | vetted |
| `GLRLM_LRLGLE` | 3.14448299246 | 3.14448299246 | 0.0e+00 | vetted |
| `GLRLM_RE` | 2.16720038143 | 2.16955079756 | 1.1e-03 | vetted (log-based band) |
| `GLRLM_RLN` | 6.12286374777 | 6.12286374777 | 1.5e-16 | vetted |
| `GLRLM_RLNN` | 0.491741380914 | 0.491741380914 | 1.1e-16 | vetted |
| `GLRLM_RP` | 0.627099458204 | 0.627099458204 | 1.8e-16 | vetted |
| `GLRLM_RV` | 0.761475060921 | 0.761475060921 | 0.0e+00 | vetted |
| `GLRLM_SRE` | 0.64062435454 | 0.64062435454 | 1.7e-16 | vetted |
| `GLRLM_SRHGLE` | 8.57313676528 | 8.57313676528 | 0.0e+00 | vetted |
| `GLRLM_SRLGLE` | 0.293965846778 | 0.293965846778 | 0.0e+00 | vetted |

## Why run entropy is not exact

`GLRLM_RE` is the family's only sum over logarithms - `glrlm.cpp` contains exactly one `log` call,
at the run-entropy accumulation - and Nyxus evaluates it with `fast_log10` plus an `EPSILON` guard
against `log(0)`, where both reference tools use the library `log`.

`fast_log10` (`src/nyx/helpers/helpers.h`) is not a rounding-level approximation: it casts the
double argument **down to float**, then approximates `log2` with the two-term polynomial
`a*(x-1)^2 + b*(x-1)` over a reduced range of [0.75, 1.5). A relative error of order 1e-3 is what
that construction costs, which is the size of the miss observed. So the explanation is quantitative,
not just plausible: it is the only feature of the sixteen that misses, it misses against both tools
by the same 1.1e-3, and the same signature appears in the GLCM family's entropy features.

It is an accuracy choice in Nyxus, not a definitional disagreement, so it is asserted at `rel=5e-3`
- about 4.5x the measured deviation - and the other fifteen at `rel=1e-9`. Worth recording that the
deviation is **avoidable**: this is one line, and evaluating it with the library `log` would put
`GLRLM_RE` on the tools to double precision like its fifteen siblings. That is a behaviour change to
a shipped feature, so it is noted here rather than made in a vetting PR.

## What this report does and does not establish

The in-tree goldens were emitted by the generator named above, so "golden == fresh run" only shows
the pin is reproducible. The vetting claim rests on the **Nyxus vs tool** columns: two independent
implementations of the same published definitions, at the same configuration, on the same pixels.
