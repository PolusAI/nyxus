# 2D first-order config matrix

The 2D first-order family has one effective setting axis: histogram bin count, read by entropy and
uniformity. Percentiles use Nyxus' fixed 100-bin interpolated CDF; the remaining statistics operate
on the ROI pixel vector. Slide range is part of the MATLAB recipe because
`COVERED_IMAGE_INTENSITY_RANGE` reads it.

| Config point | Nyxus settings | Oracle mapping | Verdict |
|---|---|---|---|
| `firstorder2d.matlab_native` | `bench_irregular13x18_intensity`, defaults; slide range 0..65535 for the slide-relative assertion | MATLAB R2026a native built-ins; derived values use only defining arithmetic over built-ins | **VALID** for 31 features at `rel=1e-3`, or `rel=3e-2` for the percentile group |
| `firstorder.pyradiomics_default` | `bench_irregular13x18_intensity`, `GREYDEPTH=64`, `IBSI=false` | PyRadiomics 3.0.1, `binCount=64`, 2D, spacing 1, label 1 | **VALID** for 18 features at the measured bands in the PyRadiomics report |
| `firstorder2d.ibsi_digital_phantom` | four-slice IBSI phantom as one ROI, default settings | published IBSI reference values | **VALID** for 12 live assertions at `rel=1e-2` |
| `firstorder2d.regression_default` | `bench_irregular13x18_intensity`, defaults | native MATLAB definitions differ | **VALID-PRODUCTION-ONLY** for `MEDIAN_ABSOLUTE_DEVIATION` and `ROBUST_MEAN` |
| `firstorder2d.regression_greydepth20` | `bench_irregular13x18_intensity`, `GREYDEPTH=20`, `IBSI=false` | no oracle at this production histogram point | **VALID-PRODUCTION-ONLY** for `ENTROPY`; its 64-bin point is vetted by PyRadiomics |

## Feature disposition

- MATLAB: 31 features in `test_2d_firstorder_matlab.h`.
- PyRadiomics: 18 features, including the only oracle assertion for `UNIFORMITY` and
  `ROBUST_MEAN_ABSOLUTE_DEVIATION`.
- IBSI: 12 live assertions on the published digital phantom.
- Regression-only: `MEDIAN_ABSOLUTE_DEVIATION` and `ROBUST_MEAN`.

There are no invalid production settings in this five-point matrix. The two regression recipes
record genuine semantic or configuration gaps and establish no oracle vetting.
