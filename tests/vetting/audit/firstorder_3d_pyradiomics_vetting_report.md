# 3D first-order vs the PyRadiomics oracle — vetting report

Every golden in `tests/test_3d_firstorder_pyradiomics.h` compared against a fresh PyRadiomics run,
and the Nyxus-vs-oracle residual measured for each so the assertion band can be set from it. These
seventeen assertions were already registered and passing before this pass; what had never been
checked is whether the numbers came from the tool and how close the agreement actually is.

## Reproduction

| | |
|---|---|
| generator | `tests/vetting/oracles/gen_firstorder3d_pyradiomics.py` |
| oracle | PyRadiomics 3.0.1 on SimpleITK 2.3.1, Python 3.8.20, NumPy 1.23.5 |
| fixture | `tests/data/nifti/compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1 |
| recipe | `firstorder.pyradiomics.bincount20` — `binCount 20`, `label 1`, no resampling, no weighting |

```
python gen_firstorder3d_pyradiomics.py
```

PyRadiomics computes first-order features on the original intensities; only `Entropy` and
`Uniformity` read the discretized histogram, which is what `binCount 20` sets and what the Nyxus
`GREYDEPTH = -20` setting matches.

## The pins are genuine

All seventeen reproduce **exactly** — `rel = 0.000e+00` on every one, against a fresh run. Unlike the
MATLAB table, these were pinned at full precision from a real invocation of the tool the registry
names, and nothing has rotted since. The generator's reverse check is clean: PyRadiomics produces no
mapped first-order feature the header leaves unpinned, and `TotalEnergy` is recorded as deliberately
unmapped.

Mechanical range and identity checks over the pin set, all passing:

| check | value |
|---|---|
| `3UNIFORMITY` in [0,1] | 0.1003710937 |
| `3ENTROPY` ≥ 0 | 3.593829137 |
| `3RANGE == 3MAX − 3MIN` | 441 |
| `3MIN ≤ 3P10 ≤ 3MEDIAN ≤ 3P90 ≤ 3MAX` | 442 |
| `3ROOT_MEAN_SQUARED² == 3MEAN² + 3VARIANCE` | 448.4583107 |
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION ≤ 3MEAN_ABSOLUTE_DEVIATION` | 33.49484999 |

## What the band was hiding

Every assertion used `agrees_gt(..., 10.)`. The third argument is a divisor, so that is a **±10%
band**. Measuring Nyxus against each pin gives the real picture:

**Twelve agree to better than 1e-12** — `3ENERGY`, `3ENTROPY`, `3KURTOSIS`, `3MAX`,
`3MEAN_ABSOLUTE_DEVIATION`, `3MEAN`, `3MEDIAN`, `3MIN`, `3RANGE`, `3ROOT_MEAN_SQUARED`,
`3SKEWNESS`, `3UNIFORMITY`. For these the band was eight orders of magnitude too loose.

**Five diverge, for two distinct reasons:**

| feature | Nyxus | PyRadiomics | rel | cause |
|---|---:|---:|---:|---|
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | 33.3279 | 33.4948499916 | 4.99e-3 | binned estimator |
| `3P10` | 362.675 | 362 | 1.86e-3 | binned estimator |
| `3P90` | 527.438 | 527 | 8.30e-4 | binned estimator |
| `3INTERQUARTILE_RANGE` | 78.9528 | 79 | 5.98e-4 | binned estimator |
| `3VARIANCE` | 4197.79 | 4196.911126692708 | 2.08e-4 | **biased vs unbiased** |

### The percentile group

`3P10`, `3P90`, `3INTERQUARTILE_RANGE` and `3ROBUST_MEAN_ABSOLUTE_DEVIATION` all come from
`TrivialHistogram`'s 100-bin interpolated estimator, while PyRadiomics uses NumPy percentiles on the
sorted array. The residual is the estimator's approximation error, not float noise, and it is the
same cause as the divergence measured against MATLAB `prctile` in the sibling report.

### `3VARIANCE` is not the feature PyRadiomics agrees with

The 2.08e-4 residual is exactly `n/(n−1)`: PyRadiomics `Variance` is the **biased** (N) estimator,
while Nyxus `3VARIANCE` is the **unbiased** (N−1) one. `3VARIANCE_BIASED` is the feature that
corresponds term for term. The registry credits `3VARIANCE` to `pyradiomics`, which is true only up
to that estimator difference — recorded in the row's note rather than silently absorbed, since the
band has to accommodate it either way.

## Bands now asserted

Set from the measurements above rather than from a round number.

| band | value | features | measured worst |
|---|---|---|---|
| `FO3D_PYRAD_EXACT` | rel 1e-9 | the twelve | < 1e-12 |
| `FO3D_PYRAD_BINNED` | rel 1e-2 | `3P10`, `3P90`, `3INTERQUARTILE_RANGE`, `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | 4.99e-3 |
| `FO3D_PYRAD_VARIANCE` | rel 1e-3 | `3VARIANCE` | 2.08e-4 |

The helper keeps its two-argument signature because the generic 3D coverage sweep calls it as
`assert_3d_firstorder_feature_pyradiomics(c.code, c.name)`; the band is looked up by feature name
inside it.

## Include hygiene

The header included `../src/nyx/raw_nifti.h`, which it uses no symbol from, and used `Fsettings` and
`NyxSetting` without including `feature_settings.h`, relying on a transitive include. It also called
`agrees_gt` without including the header that defines it. Corrected: `feature_settings.h` added,
`raw_nifti.h` dropped, and `roi_cache.h` replaced by `test_main_nyxus.h`, which supplies both
`agrees_gt` and the 3D workflow declarations and is named as the source in a trailing comment.

Its provenance block described a different fixture entirely — `ut_inten.nii`, label 57, and "100
grey levels, offset 1, and asymmetric cooc matrix", which is GLCM text — while the code loads
`compat_int_mri.nii` / `compat_seg_liver.nii` at label 1. Replaced with the recipe actually run.
