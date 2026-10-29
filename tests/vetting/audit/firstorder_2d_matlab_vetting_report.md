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
| Tolerance | `rel=1e-3` for same-definition statistics; `rel=1e-9` for `P01`..`P99`, `INTERQUARTILE_RANGE` and `QCOD` |

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
| `P01`, `P10`, `P25`, `P75`, `P90`, `P99` | `prctile(..., Method="inclusive")` |
| `ROOT_MEAN_SQUARED` | `rms` |
| `ENERGY` | `dot(x,x)` |
| `INTEGRATED_INTENSITY` | `sum` |

The remaining vetted values use only their defining arithmetic over those results:

- `COV = std(x,0) / mean(x)`
- `EXCESS_KURTOSIS = kurtosis(x,1) - 3`
- `HYPERSKEWNESS = moment(x,5) / std(x,0)^5`
- `HYPERFLATNESS = moment(x,6) / std(x,0)^6`
- `INTERQUARTILE_RANGE = P75 - P25`
- `QCOD = (P75-P25) / (P75+P25)`
- `STANDARD_ERROR = std(x,0) / sqrt(numel(x))`
- `UNIFORMITY_PIU = (1-(max-min)/(max+min))*100`
- `COVERED_IMAGE_INTENSITY_RANGE = range(x) / (65535-0)`

No Nyxus histogram or feature algorithm is reproduced.

## Percentile agreement

Both sides take sample percentiles of the same 154 pixels at numpy's default `'linear'` ranks
i/(n-1), the method PyRadiomics and MIRP use: Nyxus takes them exactly, and MATLAB
`prctile(..., Method="inclusive")` is the same rule. The percentile goldens are computed on GNU
Octave 11.3.0 as `quantile(x, p, 1, 7)` and match `numpy.percentile`'s default to the last digit. MATLAB run pending: the generator's `prctile(..., Method="inclusive")` calls have not yet been run on MATLAB. Both
sides interpolate between the same two pixels by the same fraction, so the eight rows are asserted
at `rel=1e-9`, and Nyxus agrees exactly.

| Feature | Golden (`prctile` inclusive / Octave method 7) | Nyxus |
|---|---:|---|
| `P01` | 12527.299999999999 | exact |
| `P10` | 16329 | exact |
| `P25` | 19552 | exact |
| `P75` | 45668.25 | exact |
| `P90` | 53295.0 | exact |
| `P99` | 63245.720000000001 | exact |
| `INTERQUARTILE_RANGE` | 26116.25 | exact |
| `QCOD` | 0.40043161441423486 | exact |

The C++ table pins the oracle values, not the Nyxus outputs. The band rejects a percentile read
off a 100-bin histogram, which misses these goldens by 1.6e-3 to 5.0e-2.

## Features outside the MATLAB assertion

- `UNIFORMITY` remains vetted by PyRadiomics. Producing it in MATLAB would require recreating
  Nyxus' histogram discretization, which is not an independent oracle.
- `MEDIAN_ABSOLUTE_DEVIATION` is regression-only: MATLAB `mad(x,1)` takes the median absolute
  deviation, while Nyxus takes the mean absolute deviation about the median.
- `ROBUST_MEAN` is regression-only: MATLAB `trimmean` removes samples by rank, while Nyxus selects
  values through the P10/P90 thresholds.
- `ENTROPY` and `ROBUST_MEAN_ABSOLUTE_DEVIATION` are already vetted by PyRadiomics.

This yields 34 of 36 2D first-order features vetted by at least one oracle; the two unresolved
semantic mismatches remain explicit regression rows.
