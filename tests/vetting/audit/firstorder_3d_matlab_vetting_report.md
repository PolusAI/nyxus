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
| Tool | MATLAB R2026a for the definitions; values re-derived on GNU Octave 11.3.0 + statistics |
| Fixture | `tests/data/nifti/phantoms/ut_inten.nii` and `ut_mask57.nii`, label 57 |
| Recipe | `firstorder3d.matlab_native` |
| Test | `TEST_NYXUS.TEST_3D_FIRSTORDER_MATLAB` |

The generator downloads the fixture from the moving `PolusAI/nyxus` `main` tree, applies the
documented reported-domain setup, and then evaluates the feature values. MATLAB is an
offline golden generator; CI consumes only the checked-in full-precision values.

The pins were re-derived when Nyxus began reporting first-order values in the volume's own
intensity domain rather than in the loader's grey levels. The fixture's volume minimum is
`-1024`, so the loader offsets by that and the intensity family adds it back: every location
statistic in the table moved by exactly `-1024`, every dispersion statistic is unchanged, and
`3COV`, `3QCOD`, `3UNIFORMITY_PIU` and `3ROOT_MEAN_SQUARED` were recomputed from their moved
parts. The re-derivation ran on GNU Octave 11.3.0 with the statistics package, over the same
built-ins the MATLAB generator names, and reproduced all 29 previous MATLAB values to 13
significant digits on the old domain first as a control.

Twenty-one features use a matching MATLAB function directly: `sum`, `kurtosis`, `max`, `mean`,
`mad`, `median`, `min`, `mode`, `prctile(..., Method="inclusive")`, `range`, `rms`, `skewness`,
`std`, or `var`. Eight use only the feature's defining arithmetic over those built-ins:

- `3COV`: `std / mean`
- `3INTERQUARTILE_RANGE`: `P75 - P25`
- `3STANDARD_ERROR`: `std / sqrt(n)`
- `3EXCESS_KURTOSIS`: `kurtosis - 3`
- `3QCOD`: `(P75 - P25) / (P75 + P25)`
- `3UNIFORMITY_PIU`: the PIU expression over `min` and `max`
- `3HYPERSKEWNESS` and `3HYPERFLATNESS`: standardized fifth and sixth central moments

No Nyxus percentile, robust-window, entropy, or uniformity implementation is reproduced
inside the MATLAB generator.

## Tolerances

| Group | Tolerance | Reason |
|---|---:|---|
| Same-definition native statistics | `rel=1e-3` | SPEC §7 tier for the same statistic on the same integer voxel vector |
| Percentiles, `3INTERQUARTILE_RANGE` and `3QCOD` | `rel=1e-9` | MATLAB `prctile(..., Method="inclusive")` and Nyxus take the same numpy-linear ranks; agreement to rounding |
| Zero reference (`3MIN`, `3UNIFORMITY_PIU`) | `1e-3` of the voxel range | a zero reference has no relative error to take |

Both sides take sample percentiles of the same voxel vector at numpy's default `'linear'` ranks
i/(n-1), the method PyRadiomics and MIRP use: Nyxus takes them exactly, and MATLAB
`prctile(..., Method="inclusive")` is the same rule. The percentile goldens are computed on GNU
Octave 11.3.0 as `quantile(x, p, 1, 7)` and match `numpy.percentile`'s default to the last digit. MATLAB run pending: the generator's `prctile(..., Method="inclusive")` calls have not yet been run on MATLAB. Both
sides interpolate between the same two voxels by the same fraction, so the group is compared
relatively, like every other nonzero reference.

| Feature | Golden (`prctile` inclusive / Octave method 7) | Nyxus |
|---|---:|---|
| `3P01` | 13 | exact |
| `3P10` | 164 | exact |
| `3P25` | 445 | exact |
| `3P75` | 1463.25 | exact |
| `3P90` | 1784 | exact |
| `3P99` | 1978 | exact |
| `3INTERQUARTILE_RANGE` | 1018.25 | exact |
| `3QCOD` | 0.53360408751473865 | exact |

The band rejects a percentile read off a 100-bin histogram, which misses these goldens by 1.3e-4
to 1.8e-1.

## Deliberate oracle boundary

Four features absent from the MATLAB table retain native PyRadiomics coverage from PR 449:
`3ENERGY`, `3ENTROPY`, `3ROBUST_MEAN_ABSOLUTE_DEVIATION`, and `3UNIFORMITY`.
`3ROBUST_MEAN_ABSOLUTE_DEVIATION` and the percentile group are also asserted against their
hand-worked closed form in `test_3d_firstorder_analytic.h`.

Three features remain regression-only:

| Feature | Reason |
|---|---|
| `3COVERED_IMAGE_INTENSITY_RANGE` | Nyxus-specific slide/ROI ratio with no matching external feature |
| `3MEDIAN_ABSOLUTE_DEVIATION` | MATLAB `mad(x,1)` takes the median absolute deviation; Nyxus averages deviations about the median |
| `3ROBUST_MEAN` | MATLAB `trimmean` trims by rank; Nyxus selects values through the P10/P90 thresholds |

The distinction is intentional: the regression pins preserve current Nyxus behavior but establish
no external correctness claim.

## What was removed from the Octave path

The PR 449 Octave generator formed a P10/P90 subset and evaluated robust statistics over it, and it
constructed equal-width bins before applying the entropy and uniformity equations. It also emitted
Nyxus's mean-deviation-about-the-median formula beside Octave's different native `mad(x,1)` result.
Those calculations can be useful independent implementations, but they are not direct MATLAB or
Octave feature calls. The Python wrapper, reconstructed claims, and stale 35-feature audit are
therefore removed rather than relabeled as native MATLAB evidence.
