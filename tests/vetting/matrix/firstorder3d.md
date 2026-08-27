# 3D first-order config matrix

The 3D first-order family has one effective configuration axis: the custom histogram bin count
read by entropy and uniformity. Percentiles use Nyxus's fixed 100-bin interpolated CDF; the
remaining statistics operate on the ROI voxel vector. Fixture and loader domain are part of each
recipe because absolute-intensity features depend on them.

| Config point | Nyxus settings | Oracle mapping | Verdict |
|---|---|---|---|
| `firstorder3d.matlab_native` | `bench_ut57_3d`, label 57, default settings and float-NIfTI loader domain | MATLAB R2026a native built-ins; derived values use only defining arithmetic over built-ins | **VALID** for 29 features at `rel=1e-3`, or `rel=1e-2` for the percentile group |
| `firstorder3d.pyradiomics_bincount20` | `bench_compat_liver_3d`, label 1, `GREYDEPTH=-20` | PyRadiomics 3.0.1, `binCount=20`, no resampling or weighting | **VALID** for 17 features at the measured bands in the PyRadiomics report |
| `firstorder3d.regression_ut_phantom` | `bench_ut57_3d`, label 57, default settings | No equivalent native external feature | **VALID-PRODUCTION-ONLY** for `3COVERED_IMAGE_INTENSITY_RANGE`, `3MEDIAN_ABSOLUTE_DEVIATION`, and `3ROBUST_MEAN` |

## Feature disposition

- MATLAB-only: `3COV`, `3EXCESS_KURTOSIS`, `3HYPERFLATNESS`, `3HYPERSKEWNESS`,
  `3INTEGRATED_INTENSITY`, `3MODE`, `3P01`, `3P25`, `3P75`, `3P99`, `3QCOD`,
  `3STANDARD_DEVIATION`, `3STANDARD_DEVIATION_BIASED`, `3STANDARD_ERROR`,
  `3UNIFORMITY_PIU`, `3VARIANCE_BIASED`.
- PyRadiomics-only: `3ENERGY`, `3ENTROPY`, `3ROBUST_MEAN_ABSOLUTE_DEVIATION`, `3UNIFORMITY`.
- Both oracles: `3INTERQUARTILE_RANGE`, `3KURTOSIS`, `3MAX`, `3MEAN`,
  `3MEAN_ABSOLUTE_DEVIATION`, `3MEDIAN`, `3MIN`, `3P10`, `3P90`, `3RANGE`,
  `3ROOT_MEAN_SQUARED`, `3SKEWNESS`, `3VARIANCE`.
- Regression-only: `3COVERED_IMAGE_INTENSITY_RANGE`, `3MEDIAN_ABSOLUTE_DEVIATION`,
  `3ROBUST_MEAN`.

There are no invalid production settings in this three-point matrix. The regression point exists
to record semantic mismatches honestly; it is not an oracle claim.
