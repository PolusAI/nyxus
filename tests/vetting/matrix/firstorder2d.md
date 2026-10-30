# 2D first-order config matrix

The 2D first-order family has one effective setting axis: histogram bin count, read by entropy and
uniformity. Percentiles are numpy's default linear sample percentiles of the ROI pixel vector, on
which the remaining statistics operate too. Slide range is part of the MATLAB recipe because
`COVERED_IMAGE_INTENSITY_RANGE` reads it.

| Config point | Nyxus settings | Oracle mapping | Verdict |
|---|---|---|---|
| `firstorder2d.matlab_native` | `bench_irregular13x18_intensity`, defaults; slide range 0..65535 for the slide-relative assertion | MATLAB R2026a native built-ins; derived values use only defining arithmetic over built-ins | **VALID** for 31 features at `rel=1e-3`, or `rel=1e-9` for the percentile group, where `prctile(..., Method="inclusive")` takes numpy's linear ranks as Nyxus does |
| `firstorder.pyradiomics_default` | `bench_irregular13x18_intensity`, `GREYDEPTH=64`, `IBSI=false` | PyRadiomics 3.0.1, `binCount=64`, 2D, spacing 1, label 1 | **VALID** for 18 features at the measured bands in the PyRadiomics report |
| `firstorder2d.ibsi_digital_phantom` | four-slice IBSI phantom as one ROI, default settings | published IBSI reference values | **VALID** for 13 live assertions at `rel=1e-2` |
| `firstorder2d.percentiles_analytic` | `bench_percentile_hand_samples`, defaults | closed form of numpy's linear percentile rule, worked by hand | **VALID** for 9 percentile-derived features at `abs=1e-12` |
| `firstorder2d.regression_default` | `bench_irregular13x18_intensity`, defaults | native MATLAB definitions differ | **VALID-PRODUCTION-ONLY** for `MEDIAN_ABSOLUTE_DEVIATION` and `ROBUST_MEAN` |
| `firstorder2d.regression_greydepth20` | `bench_irregular13x18_intensity`, `GREYDEPTH=20`, `IBSI=false` | no oracle at this production histogram point | **VALID-PRODUCTION-ONLY** for `ENTROPY`; its 64-bin point is vetted by PyRadiomics |

## Feature disposition

- MATLAB: 31 features in `test_2d_firstorder_matlab.h`.
- PyRadiomics: 18 features, including the only oracle assertion for `UNIFORMITY`.
- IBSI: 13 live assertions on the published digital phantom.
- Analytic: `P01`, `P10`, `P25`, `P75`, `P90`, `P99`, `INTERQUARTILE_RANGE`, `QCOD` and
  `ROBUST_MEAN_ABSOLUTE_DEVIATION` in `test_2d_firstorder_analytic.h`.
- Regression-only: `MEDIAN_ABSOLUTE_DEVIATION` and `ROBUST_MEAN`.

There are no invalid production settings in this six-point matrix. The two regression recipes
record genuine semantic or configuration gaps and establish no oracle vetting.
