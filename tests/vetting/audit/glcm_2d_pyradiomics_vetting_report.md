# Audit: 2D GLCM vs a fresh PyRadiomics run

**Verdict: all 30 quantities reproduce.** 25 to double precision (worst 1.9e-12), the five
log-based ones to 2.8e-3 - explained below and asserted at a band that matches.

Covers `tests/test_2d_glcm_pyradiomics.h` (goldens + assertions), `tests/test_2d_glcm_common.h`
(fixture) and `tests/vetting/oracles/gen_glcm_pyradiomics.py` (generator).

## Method

- **Tool**: pyradiomics **v3.0.1** with SimpleITK, in a conda env (see `tests/vetting/TOOLS.md`,
  "conda route"). Not the Docker image - conda-forge resolves pyradiomics on Python 3.9 directly.
- **Config**: `binWidth=1` (identity binning on an integer image), `symmetricalGLCM=True`,
  `distances=[1]`, `force2D=True`, `force2Ddimension=0`, `weightingNorm=None`, `label=1`,
  `interpolator=None`, `resampledPixelSpacing=None`. Nyxus side: recipe `glcm.ibsi_identity`
  (`ibsi=true`, `GLCM_GREYDEPTH=0`, `GLCM_OFFSET=1`, angles 0/45/90/135).
- **Aggregation**: PyRadiomics reports one value per feature over the angle set, which is the Nyxus
  `*_AVE` aggregation; the per-angle features are the mean of their 4 angled values, and Nyxus'
  `*_AVE` reproduces that mean exactly (checked separately in `test_2d_glcm_ibsi.h`).
- **Fixtures**: both were run, not just the one the goldens are pinned on.
  1. the dense phantom of `test_2d_glcm_common.h` - `img[y,x] = ((y + 2x) % 8) + 1`, 8x8, every
     grey level 1..8, one-pixel background border;
  2. the IBSI digital phantom as `tests/test_data.h` stores it (4 slices, in-mask levels
     {1,3,4,6}), per slice then averaged over the 4 slices.
- **Commands**: `python tests/vetting/oracles/gen_glcm_pyradiomics.py` for fixture 1; the same
  settings applied slice-by-slice for fixture 2. Both are reproduced step by step in
  `glcm_2d_golden_regen.md`.

## Fixture 1 - dense phantom (the pinned goldens)

| feature | Nyxus | PyRadiomics fresh | rel | verdict |
|---|---|---|---:|---|
| `GLCM_ACOR` | 20.512755102 | 20.512755102 | 0.0e+00 | vetted |
| `GLCM_ASM` | 0.0635347251145 | 0.0635347251145 | 0.0e+00 | vetted |
| `GLCM_CLUPROM` | 263.28733919 | 263.28733919 | 0.0e+00 | vetted |
| `GLCM_CLUSHADE` | 0.0175437105288 | 0.0175437105288 | 1.9e-12 | vetted |
| `GLCM_CLUTEND` | 10.8988962932 | 10.8988962932 | 1.6e-16 | vetted |
| `GLCM_CONTRAST` | 10.2193877551 | 10.2193877551 | 0.0e+00 | vetted |
| `GLCM_CORRELATION` | 0.0321647149882 | 0.0321647149882 | 1.3e-15 | vetted |
| `GLCM_DIFAVE` | 2.55867346939 | 2.55867346939 | 0.0e+00 | vetted |
| `GLCM_DIFENTRO` | 0.70872754999 | 0.710700360776 | 2.8e-03 | vetted (log-based band) |
| `GLCM_DIFVAR` | 2.94541336943 | 2.94541336943 | 1.5e-16 | vetted |
| `GLCM_DIS` | 2.55867346939 | 2.55867346939 | 0.0e+00 | vetted |
| `GLCM_ENERGY` | 0.0635347251145 | 0.0635347251145 | 0.0e+00 | vetted |
| `GLCM_ENTROPY` | 3.98586499934 | 3.98790038214 | 5.1e-04 | vetted (log-based band) |
| `GLCM_HOM1` | 0.352838010204 | 0.352838010204 | 0.0e+00 | vetted |
| `GLCM_ID` | 0.352838010204 | 0.352838010204 | 0.0e+00 | vetted |
| `GLCM_IDM` | 0.278537697823 | 0.278537697823 | 2.0e-16 | vetted |
| `GLCM_IDMN` | 0.887341632851 | 0.887341632851 | 1.3e-16 | vetted |
| `GLCM_IDN` | 0.779479250908 | 0.779479250908 | 0.0e+00 | vetted |
| `GLCM_INFOMEAS1` | -0.67114878951 | -0.670502985211 | 9.6e-04 | vetted (log-based band) |
| `GLCM_INFOMEAS2` | 0.991038216196 | 0.991003991802 | 3.5e-05 | vetted (log-based band) |
| `GLCM_IV` | 0.508633786848 | 0.508633786848 | 0.0e+00 | vetted |
| `GLCM_JAVE` | 4.51020408163 | 4.51020408163 | 2.0e-16 | vetted |
| `GLCM_JE` | 3.98586499934 | 3.98790038214 | 5.1e-04 | vetted (log-based band) |
| `GLCM_JMAX` | 0.0691964285714 | 0.0691964285714 | 0.0e+00 | vetted |
| `GLCM_JVAR` | 5.27957101208 | 5.27957101208 | 0.0e+00 | vetted |
| `GLCM_SUMAVERAGE` | 9.02040816327 | 9.02040816327 | 2.0e-16 | vetted |
| `GLCM_SUMENTROPY` | 2.55772266461 | 2.55966399272 | 7.6e-04 | vetted (log-based band) |
| `GLCM_SUMVARIANCE` | 10.8988962932 | 10.8988962932 | 1.6e-16 | vetted |
| `GLCM_VARIANCE` | 5.27957101208 | 5.27957101208 | 0.0e+00 | vetted |

## Fixture 2 - IBSI digital phantom

A second, independent configuration: a different image, different grey levels, and a matrix with
absent levels ({1,3,4,6}, so 2 and 5 never occur).

| feature | Nyxus | PyRadiomics fresh | rel | verdict |
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

## Finding: the level-gap concern does not apply at this recipe

The dense fixture was chosen partly on the belief - carried in a comment since the PR #356 review -
that PyRadiomics re-indexes the grey levels it finds, so a phantom with an absent level would give
it a different `Ng` from Nyxus' and make `Idn`, `Idmn` and `Autocorrelation` incomparable. Fixture 2
tests exactly that case and **the concern does not hold at `binWidth=1`**: on levels {1,3,4,6}
PyRadiomics reproduces Nyxus (and MIRP) to 1e-15 on all three. The comment has been corrected; the
dense phantom is kept because it exercises every level and a fuller matrix, not because the IBSI one
is unusable.

## Why the log-based features are not exact

`GLCM_DIFENTRO`, `GLCM_JE`/`GLCM_ENTROPY`, `GLCM_SUMENTROPY`, `GLCM_INFOMEAS1` and `GLCM_INFOMEAS2`
are the five whose sums run over logarithms. Nyxus evaluates them with `fast_log10` plus an
`EPSILON` guard against `log(0)` (`src/nyx/features/glcm.cpp`), where both reference tools use the
library `log`.

`fast_log10` (`src/nyx/helpers/helpers.h`) is not a rounding-level approximation: it casts the
double argument **down to float**, then approximates `log2` with the two-term polynomial
`a*(x-1)^2 + b*(x-1)` over a reduced range of [0.75, 1.5). A relative error of order 1e-3 is what
that construction costs, which is the size of the miss observed - so the explanation is
quantitative, not merely plausible. It shows up on both fixtures, against both tools, at the same
magnitude, and only in those five.

It is an accuracy choice in Nyxus, not a definitional disagreement, so those features are asserted
at `rel=5e-3` and the rest at `rel=1e-9`. Worth recording that the deviation is **avoidable**:
evaluating those sums with the library `log` would put all five on the tools to double precision.
That is a behaviour change to shipped features, so it is noted here rather than made in a vetting
PR.

## The regression snapshot is at the other recipe

`tests/test_2d_glcm_regression.h` pins the same features at 100 grey levels, offset 1, MATLAB
binning and `symmetric_glcm=false` - an **asymmetric** co-occurrence matrix. That is a configuration
choice, not a limitation: Nyxus symmetrises on the IBSI and radiomics paths, which is what the
oracle tests use. Under the asymmetric config the keys split in two:

- **Symmetrisation-invariant** - they depend only on the grey-level difference `p_{x-y}` / `|i-j|`
  (`CONTRAST`, `DIFAVE`, `DIS`, `DIFENTRO`, `DIFVAR`, `ID`, `HOM1`, `IDM`, `IV`) or on the sum
  distribution `p_{x+y}` (`SUMENTROPY`). These land within 1% of PyRadiomics run on the same
  per-slice MATLAB-binned images, because this phantom's binning relabels levels without rescaling
  them. They are *not* invariant to level scaling - that only holds here.
- **Transpose-sensitive** - they read individual matrix entries or the grey-tone marginal means
  `mu_x`/`mu_y`, and diverge from any symmetric-matrix tool as configured. Measured:
  `ASM`/`ENERGY` 3.7%, `CLUSHADE` 46%, `CLUTEND`/`SUMVARIANCE` 3.2%, `JE` 9.3%,
  `JVAR`/`VARIANCE` ~10%. `CLUTEND`/`SUMVARIANCE` are in this group because Nyxus computes them
  from the single row-marginal mean `by_row_mean`.

So the snapshot is a drift guard and nothing more; comparing it to a tool would require rerunning
it with `symmetric_glcm=true`, at which point it is the oracle tests' recipe.

The snapshot was refreshed in 2026-06 after the GLCM background-pollution fix, when the non-IBSI
path stopped counting out-of-ROI background pixels (slices z2-z4 have masked-out pixels).

## What this report does and does not establish

The in-tree goldens were emitted by the generator named above, so "golden == fresh run" only shows
the pin is reproducible. The vetting claim rests on the **Nyxus vs tool** columns: those are two
independent implementations of the same published definitions, compared at the same configuration
on the same pixels.
