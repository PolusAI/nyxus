# 3D first-order MATLAB R2026a vetting report

This report supersedes the GNU Octave audit merged in PR 449. That work correctly made the 3D
first-order family runnable and retained useful PyRadiomics evidence, but its Octave generator also
manually reconstructed several feature definitions. Those reconstructed values are not MATLAB
built-in oracle evidence under the strict convention used here.

## Current accounting

| | Count |
|---|---:|
| Nyxus 3D first-order features | 36 |
| Vetted by at least one oracle | 33 |
| Regression-only | 3 |
| MATLAB R2026a assertions | 29 |
| PyRadiomics 3.0.1 assertions | 17 |
| Features covered by both tools | 13 |

The authoritative feature × config × oracle records are in `tests/vetting/oracle_coverage.csv`.
The MATLAB pins and their owning assertion are in `tests/test_3d_firstorder_matlab.h`.

## MATLAB provenance

| | |
|---|---|
| Generator | `tests/vetting/oracles/gen_firstorder3d_matlab.m` |
| Tool | MATLAB R2026a |
| Fixture | `tests/data/nifti/phantoms/ut_inten.nii` and `ut_mask57.nii`, label 57 |
| Recipe | `firstorder3d.matlab_native` |
| Test | `TEST_NYXUS.TEST_3D_FIRSTORDER_MATLAB` |

The generator downloads the fixture from the moving `PolusAI/nyxus` `main` tree, applies the
documented float-NIfTI loader-domain setup, and then evaluates the feature values. MATLAB is an
offline golden generator; CI consumes only the checked-in full-precision values.

Twenty-two features use a matching MATLAB function directly: `sum`, `iqr`, `kurtosis`, `max`,
`mean`, `mad`, `median`, `min`, `mode`, `prctile`, `range`, `rms`, `skewness`, `std`, or `var`.
Seven use only the feature's defining arithmetic over those built-ins:

- `3COV`: `std / mean`
- `3STANDARD_ERROR`: `std / sqrt(n)`
- `3EXCESS_KURTOSIS`: `kurtosis - 3`
- `3QCOD`: `(P75 - P25) / (P75 + P25)`
- `3UNIFORMITY_PIU`: the PIU expression over `min` and `max`
- `3HYPERSKEWNESS` and `3HYPERFLATNESS`: standardized fifth and sixth central moments

No Nyxus histogram, percentile, robust-window, entropy, or uniformity implementation is reproduced
inside the MATLAB generator.

## Tolerances

| Group | Tolerance | Reason |
|---|---:|---|
| Same-definition native statistics | `rel=1e-3` | SPEC §7 tier for the same statistic on the same integer voxel vector |
| Percentile-derived statistics | `rel=1e-2` | MATLAB sample percentiles versus Nyxus's fixed 100-bin interpolated CDF |

The worst measured MATLAB residual is `2.30e-3` on `3P01` (`Nyxus 1039.3829596413`, MATLAB
`1037`). It is inside the common one-percent percentile band. The tolerance records agreement; it
does not require bit identity between two different percentile estimators.

## Deliberate oracle boundary

Four features absent from the MATLAB table retain native PyRadiomics coverage from PR 449:
`3ENERGY`, `3ENTROPY`, `3ROBUST_MEAN_ABSOLUTE_DEVIATION`, and `3UNIFORMITY`.

Three features remain regression-only:

| Feature | Reason |
|---|---|
| `3COVERED_IMAGE_INTENSITY_RANGE` | Nyxus-specific slide/ROI ratio with no matching external feature |
| `3MEDIAN_ABSOLUTE_DEVIATION` | MATLAB `mad(x,1)` takes the median absolute deviation; Nyxus averages deviations about the median |
| `3ROBUST_MEAN` | MATLAB `trimmean` trims by rank; Nyxus selects values through histogram-derived P10/P90 thresholds |

The distinction is intentional: the regression pins preserve current Nyxus behavior but establish
no external correctness claim.

## What was removed from the Octave path

The PR 449 Octave generator formed a P10/P90 subset and evaluated robust statistics over it, and it
constructed equal-width bins before applying the entropy and uniformity equations. It also emitted
Nyxus's mean-deviation-about-the-median formula beside Octave's different native `mad(x,1)` result.
Those calculations can be useful independent implementations, but they are not direct MATLAB or
Octave feature calls. The Python wrapper, reconstructed claims, and stale 35-feature audit are
therefore removed rather than relabeled as native MATLAB evidence.
