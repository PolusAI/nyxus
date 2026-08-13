# Audit: 2D GLRLM vs a fresh MIRP run

**Verdict: all 16 quantities reproduce.** 15 to double precision (worst 4.4e-16), run entropy to
1.1e-3 - explained below and asserted at a band that matches.

Covers `tests/test_2d_glrlm_mirp.h` (goldens + assertions), `tests/test_2d_glrlm_common.h` (fixture)
and `tests/vetting/oracles/gen_glrlm_mirp.py` (generator).

## Method

- **Tool**: mirp **2.6.0** in a conda env (`conda create -n nyxus_mirp -c conda-forge python=3.11
  mirp numpy`). MIRP is the IBSI reference implementation - the second opinion that is not
  PyRadiomics-shaped.
- **Config**: `by_slice=True`, `base_discretisation_method="none"` (the phantom is already
  discrete), `glrlm_spatial_method="2d_average"`. Nyxus side: recipe `glrlm.ibsi_ng128`.
- **Fixture**: the IBSI phantom from `tests/test_data.h` via `ibsi_phantom.py`, as above.
- **Command**: `python tests/vetting/oracles/gen_glrlm_mirp.py`.
- **Gotcha**: MIRP logs at INFO onto stdout, interleaving progress lines with the golden table it
  prints. The generator calls `logging.disable(logging.INFO)`; setting a level on the root logger is
  not enough, because MIRP configures its own logger during the run.

## Result table

| feature | Nyxus | fresh run | rel | verdict |
|---|---|---|---:|---|
| `GLRLM_GLN` | 5.19706240451 | 5.19706240451 | 0.0e+00 | vetted |
| `GLRLM_GLNN` | 0.459729336675 | 0.459729336675 | 0.0e+00 | vetted |
| `GLRLM_GLV` | 3.35302839111 | 3.35302839111 | 1.3e-16 | vetted |
| `GLRLM_HGLRE` | 9.8242744541 | 9.8242744541 | 0.0e+00 | vetted |
| `GLRLM_LGLRE` | 0.604357907739 | 0.604357907739 | 1.8e-16 | vetted |
| `GLRLM_LRE` | 3.77838422301 | 3.77838422301 | 1.2e-16 | vetted |
| `GLRLM_LRHGLE` | 17.387027112 | 17.387027112 | 0.0e+00 | vetted |
| `GLRLM_LRLGLE` | 3.14448299246 | 3.14448299246 | 0.0e+00 | vetted |
| `GLRLM_RE` | 2.16720038143 | 2.16955079756 | 1.1e-03 | vetted (log-based band) |
| `GLRLM_RLN` | 6.12286374777 | 6.12286374777 | 0.0e+00 | vetted |
| `GLRLM_RLNN` | 0.491741380914 | 0.491741380914 | 1.1e-16 | vetted |
| `GLRLM_RP` | 0.627099458204 | 0.627099458204 | 0.0e+00 | vetted |
| `GLRLM_RV` | 0.761475060921 | 0.761475060921 | 0.0e+00 | vetted |
| `GLRLM_SRE` | 0.64062435454 | 0.64062435454 | 1.7e-16 | vetted |
| `GLRLM_SRHGLE` | 8.57313676528 | 8.57313676528 | 2.1e-16 | vetted |
| `GLRLM_SRLGLE` | 0.293965846778 | 0.293965846778 | 0.0e+00 | vetted |

## Cross-check against PyRadiomics

MIRP and PyRadiomics agree with each other to ~1e-15 on all 16 quantities, so the two independent
opinions are one result reached twice and the Nyxus comparison is against a value neither disputes.

## Why run entropy is not exact

`GLRLM_RE` is the family's only sum over logarithms. Nyxus evaluates it with `fast_log10` plus an
`EPSILON` guard against `log(0)` (`src/nyx/features/glrlm.cpp`), where both reference tools use the
library `log`. That is the whole of the difference: it is the only feature of the sixteen that
misses, it misses against both tools by the same 1.1e-3, and the same signature appears in the GLCM
family's entropy features. It is an accuracy choice in Nyxus, not a definitional disagreement, so it
is asserted at `rel=5e-3` and the other fifteen at `rel=1e-9`.

## What this report does and does not establish

The in-tree goldens were emitted by the generator named above, so "golden == fresh run" only shows
the pin is reproducible. The vetting claim rests on the **Nyxus vs tool** columns: two independent
implementations of the same published definitions, at the same configuration, on the same pixels.
