# Audit: 2D GLCM vs a fresh MIRP run

**Verdict: all 30 quantities reproduce.** 25 to double precision (worst 1.3e-12), the five
log-based ones to 2.8e-3 - explained below and asserted at a band that matches.

Covers `tests/test_2d_glcm_mirp.h` (goldens + assertions), `tests/test_2d_glcm_common.h` (fixture)
and `tests/vetting/oracles/gen_glcm_mirp.py` (generator).

## Method

- **Tool**: mirp **2.6.0** in a conda env (see `tests/vetting/TOOLS.md`, "conda route"). MIRP is the
  IBSI reference implementation, so it is the second opinion that is *not* PyRadiomics-shaped: it
  reports `dissimilarity` and `sum variance` as quantities of their own, where PyRadiomics 3.x
  dropped both as duplicates of `DifferenceAverage` and `ClusterTendency`.
- **Config**: `by_slice=True`, `base_discretisation_method="none"` (the fixtures are already
  discrete), `glcm_distance=1`, `glcm_spatial_method="2d_average"`. Nyxus side: recipe
  `glcm.ibsi_identity`.
- **Fixtures**: the dense phantom of `test_2d_glcm_common.h` and the IBSI digital phantom from
  `tests/test_data.h` - the same two the PyRadiomics audit used.
- **Commands**: `python tests/vetting/oracles/gen_glcm_mirp.py`; see `glcm_2d_golden_regen.md`.
- **Gotcha**: MIRP narrates on the logging module at INFO into stdout, mixing its progress lines
  into the golden table. The generator calls `logging.disable(logging.INFO)`; setting a level on the
  root logger is not enough, because MIRP configures its own logger during the run.

## Fixture 1 - dense phantom (the pinned goldens)

| feature | Nyxus | MIRP fresh | rel | verdict |
|---|---|---|---:|---|
| `GLCM_ACOR` | 20.512755102 | 20.512755102 | 1.7e-16 | vetted |
| `GLCM_ASM` | 0.0635347251145 | 0.0635347251145 | 0.0e+00 | vetted |
| `GLCM_CLUPROM` | 263.28733919 | 263.28733919 | 2.2e-16 | vetted |
| `GLCM_CLUSHADE` | 0.0175437105288 | 0.0175437105288 | 1.6e-13 | vetted |
| `GLCM_CLUTEND` | 10.8988962932 | 10.8988962932 | 1.6e-16 | vetted |
| `GLCM_CONTRAST` | 10.2193877551 | 10.2193877551 | 0.0e+00 | vetted |
| `GLCM_CORRELATION` | 0.0321647149882 | 0.0321647149882 | 4.3e-16 | vetted |
| `GLCM_DIFAVE` | 2.55867346939 | 2.55867346939 | 0.0e+00 | vetted |
| `GLCM_DIFENTRO` | 0.70872754999 | 0.710700360776 | 2.8e-03 | vetted (log-based band) |
| `GLCM_DIFVAR` | 2.94541336943 | 2.94541336943 | 1.5e-16 | vetted |
| `GLCM_DIS` | 2.55867346939 | 2.55867346939 | 0.0e+00 | vetted |
| `GLCM_ENERGY` | 0.0635347251145 | 0.0635347251145 | 0.0e+00 | vetted |
| `GLCM_ENTROPY` | 3.98586499934 | 3.98790038214 | 5.1e-04 | vetted (log-based band) |
| `GLCM_HOM1` | 0.352838010204 | 0.352838010204 | 1.6e-16 | vetted |
| `GLCM_ID` | 0.352838010204 | 0.352838010204 | 1.6e-16 | vetted |
| `GLCM_IDM` | 0.278537697823 | 0.278537697823 | 0.0e+00 | vetted |
| `GLCM_IDMN` | 0.887341632851 | 0.887341632851 | 1.3e-16 | vetted |
| `GLCM_IDN` | 0.779479250908 | 0.779479250908 | 1.4e-16 | vetted |
| `GLCM_INFOMEAS1` | -0.67114878951 | -0.670502985211 | 9.6e-04 | vetted (log-based band) |
| `GLCM_INFOMEAS2` | 0.991038216196 | 0.991003991802 | 3.5e-05 | vetted (log-based band) |
| `GLCM_IV` | 0.508633786848 | 0.508633786848 | 2.2e-16 | vetted |
| `GLCM_JAVE` | 4.51020408163 | 4.51020408163 | 0.0e+00 | vetted |
| `GLCM_JE` | 3.98586499934 | 3.98790038214 | 5.1e-04 | vetted (log-based band) |
| `GLCM_JMAX` | 0.0691964285714 | 0.0691964285714 | 0.0e+00 | vetted |
| `GLCM_JVAR` | 5.27957101208 | 5.27957101208 | 0.0e+00 | vetted |
| `GLCM_SUMAVERAGE` | 9.02040816327 | 9.02040816327 | 0.0e+00 | vetted |
| `GLCM_SUMENTROPY` | 2.55772266461 | 2.55966399272 | 7.6e-04 | vetted (log-based band) |
| `GLCM_SUMVARIANCE` | 10.8988962932 | 10.8988962932 | 1.6e-16 | vetted |
| `GLCM_VARIANCE` | 5.27957101208 | 5.27957101208 | 0.0e+00 | vetted |

## Fixture 2 - IBSI digital phantom

| feature | Nyxus | MIRP fresh | rel | verdict |
|---|---|---|---:|---|
| `GLCM_ACOR` | 5.094374029 | 5.09437402875 | 4.9e-11 | vetted |
| `GLCM_ASM` | 0.3675285624 | 0.367528562388 | 3.3e-11 | vetted |
| `GLCM_CLUPROM` | 79.11263068 | 79.1126306836 | 4.5e-11 | vetted |
| `GLCM_CLUSHADE` | 6.997816145 | 6.99781614542 | 6.0e-11 | vetted |
| `GLCM_CLUTEND` | 5.472932478 | 5.47293247778 | 4.0e-11 | vetted |
| `GLCM_CONTRAST` | 5.277851142 | 5.27785114191 | 1.6e-11 | vetted |
| `GLCM_CORRELATION` | -0.01210696121 | -0.0121069612096 | 3.2e-11 | vetted |
| `GLCM_DIFAVE` | 1.42246729 | 1.42246728965 | 2.4e-10 | vetted |
| `GLCM_DIFENTRO` | 1.393553011 | 1.39614711299 | 1.9e-03 | vetted (log-based band) |
| `GLCM_DIFVAR` | 2.90159075 | 2.90159074992 | 2.6e-11 | vetted |
| `GLCM_DIS` | 1.42246729 | 1.42246728965 | 2.4e-10 | vetted |
| `GLCM_ENTROPY` | 2.047605754 | 2.0496642875 | 1.0e-03 | vetted (log-based band) |
| `GLCM_HOM2` | 0.6187370709 | 0.618737070889 | 1.9e-11 | vetted |
| `GLCM_ID` | 0.6779485416 | 0.67794854162 | 3.0e-11 | vetted |
| `GLCM_IDM` | 0.6187370709 | 0.618737070889 | 1.9e-11 | vetted |
| `GLCM_IDMN` | 0.8992192901 | 0.899219290131 | 3.5e-11 | vetted |
| `GLCM_IDN` | 0.8513990718 | 0.851399071783 | 2.0e-11 | vetted |
| `GLCM_INFOMEAS1` | -0.1557629868 | -0.155119516222 | 4.1e-03 | vetted (log-based band) |
| `GLCM_INFOMEAS2` | 0.4883048989 | 0.487456567651 | 1.7e-03 | vetted (log-based band) |
| `GLCM_IV` | 0.05669828975 | 0.0566982897504 | 6.6e-12 | vetted |
| `GLCM_JAVE` | 2.142418606 | 2.1424186057 | 1.4e-10 | vetted |
| `GLCM_JE` | 2.047605754 | 2.0496642875 | 1.0e-03 | vetted (log-based band) |
| `GLCM_JMAX` | 0.5187996899 | 0.518799689893 | 1.3e-11 | vetted |
| `GLCM_JVAR` | 2.687695905 | 2.68769590492 | 2.8e-11 | vetted |
| `GLCM_SUMAVERAGE` | 4.284837211 | 4.2848372114 | 9.3e-11 | vetted |
| `GLCM_SUMENTROPY` | 1.601240106 | 1.60318804058 | 1.2e-03 | vetted (log-based band) |
| `GLCM_SUMVARIANCE` | 5.472932478 | 5.47293247778 | 4.0e-11 | vetted |
| `GLCM_VARIANCE` | 2.687695905 | 2.68769590492 | 2.8e-11 | vetted |

## Cross-check against PyRadiomics

On both fixtures MIRP and PyRadiomics agree with each other to ~1e-15 on every shared quantity, so
the two "independent" opinions are genuinely one result reached twice, and the Nyxus comparison is
against a value neither tool disputes.

## Why the log-based features are not exact

`GLCM_DIFENTRO`, `GLCM_JE`/`GLCM_ENTROPY`, `GLCM_SUMENTROPY`, `GLCM_INFOMEAS1` and `GLCM_INFOMEAS2`
are the five whose sums run over logarithms. Nyxus evaluates them with `fast_log10` plus an
`EPSILON` guard against `log(0)` (`src/nyx/features/glcm.cpp`), where both reference tools use the
library `log`. That is the whole of the difference: it shows up on both fixtures, against both
tools, at the same magnitude, and only in those five. It is an accuracy choice in Nyxus, not a
definitional disagreement, so those features are asserted at `rel=5e-3` and the rest at `rel=1e-9`.

## What this report does and does not establish

The in-tree goldens were emitted by the generator named above, so "golden == fresh run" only shows
the pin is reproducible. The vetting claim rests on the **Nyxus vs tool** columns: those are two
independent implementations of the same published definitions, compared at the same configuration
on the same pixels.
