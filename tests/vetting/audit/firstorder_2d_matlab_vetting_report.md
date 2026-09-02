# 2D first-order MATLAB R2026a vetting report

This replaces the unsupported MATLAB provenance and Nyxus-derived percentile pins in
`test_2d_firstorder_matlab.h` with a licensed MATLAB R2026a run. MATLAB directly supports 31 of
the family's 36 features under the conventions below.

## Provenance

| | |
|---|---|
| Tool | MATLAB R2026a, Statistics and Machine Learning Toolbox 26.1 |
| Generator | `tests/vetting/oracles/gen_firstorder2d_matlab.m` |
| Fixture | `pixelIntensityFeaturesTestData` from `tests/test_data.h` on `PolusAI/nyxus` main |
| Recipe | `firstorder2d.matlab_native` |
| Nyxus config | default first-order settings; slide range 0..65535 for `COVERED_IMAGE_INTENSITY_RANGE` |
| Test | `test_2d_firstorder_matlab.h` |
| Tolerance | `rel=1e-3` for same-definition statistics; `rel=3e-2` for percentile-derived statistics |

Run from the repository root:

```text
matlab -batch "run('tests/vetting/oracles/gen_firstorder2d_matlab.m')"
```

The generator downloads the fixture and C++ golden table from the moving `PolusAI/nyxus` `main`
tree, checks feature names in both directions, prints MATLAB and C++ values side by side, and fails
when a value exceeds its declared tolerance.

## What MATLAB computes

| Nyxus features | MATLAB source |
|---|---|
| `MIN`, `MAX`, `RANGE`, `MEAN`, `MEDIAN`, `MODE` | matching native built-ins |
| `STANDARD_DEVIATION`, `STANDARD_DEVIATION_BIASED` | `std(x,0)`, `std(x,1)` |
| `VARIANCE`, `VARIANCE_BIASED` | `var(x,0)`, `var(x,1)` |
| `SKEWNESS`, `KURTOSIS` | `skewness(x,1)`, `kurtosis(x,1)` |
| `MEAN_ABSOLUTE_DEVIATION` | `mad(x,0)` |
| `P01`, `P10`, `P25`, `P75`, `P90`, `P99` | `prctile(..., Method="midpoint")` |
| `INTERQUARTILE_RANGE` | `iqr` |
| `ROOT_MEAN_SQUARED` | `rms` |
| `ENERGY` | `dot(x,x)` |
| `INTEGRATED_INTENSITY` | `sum` |

The remaining vetted values use only their defining arithmetic over those results:

- `COV = std(x,0) / mean(x)`
- `EXCESS_KURTOSIS = kurtosis(x,1) - 3`
- `HYPERSKEWNESS = moment(x,5) / std(x,0)^5`
- `HYPERFLATNESS = moment(x,6) / std(x,0)^6`
- `QCOD = (P75-P25) / (P75+P25)`
- `STANDARD_ERROR = std(x,0) / sqrt(numel(x))`
- `UNIFORMITY_PIU = (1-(max-min)/(max+min))*100`
- `COVERED_IMAGE_INTENSITY_RANGE = range(x) / (65535-0)`

No Nyxus histogram or feature algorithm is reproduced.

## Percentile agreement

Nyxus estimates percentiles from its fixed 100-bin interpolated CDF; MATLAB evaluates the raw
sample. These are distinct estimators of the same statistics. All eight comparisons agree within
`rel=3e-2`; the worst is `QCOD` at 2.75%.

| Feature | Nyxus | MATLAB R2026a | Relative error |
|---|---:|---:|---:|
| `P01` | 11895.3694 | 12081.4 | 1.54e-2 |
| `P10` | 16107.472 | 16329 | 1.36e-2 |
| `P25` | 19074.82583333333 | 19552 | 2.44e-2 |
| `P75` | 45801.205 | 45723 | 1.71e-3 |
| `P90` | 53381.778 | 53360.7 | 3.95e-4 |
| `P99` | 63416.7603 | 63380.96 | 5.65e-4 |
| `INTERQUARTILE_RANGE` | 26726.37916666667 | 26171 | 2.12e-2 |
| `QCOD` | 0.411960763064047 | 0.40093450785139795 | 2.75e-2 |

The C++ table pins the exact MATLAB values, not the Nyxus outputs. The tolerance expresses the
measured estimator difference; it is not a re-baseline of the oracle.

## Features outside the MATLAB assertion

- `UNIFORMITY` remains vetted by PyRadiomics. Producing it in MATLAB would require recreating
  Nyxus' histogram discretization, which is not an independent oracle.
- `MEDIAN_ABSOLUTE_DEVIATION` is regression-only: MATLAB `mad(x,1)` takes the median absolute
  deviation, while Nyxus takes the mean absolute deviation about the median.
- `ROBUST_MEAN` is regression-only: MATLAB `trimmean` removes samples by rank, while Nyxus selects
  values through histogram-derived P10/P90 thresholds.
- `ENTROPY` and `ROBUST_MEAN_ABSOLUTE_DEVIATION` are already vetted by PyRadiomics.

This yields 34 of 36 2D first-order features vetted by at least one oracle; the two unresolved
semantic mismatches remain explicit regression rows.
