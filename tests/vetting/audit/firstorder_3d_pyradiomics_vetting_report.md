# 3D first-order PyRadiomics vetting report

This report covers the non-overlapping PyRadiomics work retained from PR 449. The MATLAB/Octave
parts of that PR are not carried forward because PR 450 independently provides native MATLAB
R2026a goldens and assertions.

## Reproduction

| | |
|---|---|
| generator | `tests/vetting/oracles/gen_firstorder3d_pyradiomics.py` |
| PR 449 run | PyRadiomics 3.0.1, SimpleITK 2.3.1, Python 3.8.20, NumPy 1.23.5 |
| retained-work run | PyRadiomics 3.0.1, SimpleITK 2.5.6, Python 3.11.15, NumPy 2.4.6 |
| fixture | `tests/data/nifti/compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1 |
| recipe | `firstorder3d.pyradiomics_bincount20`: `binCount=20`, label 1, no resampling, no weighting |

Run from `tests/vetting/oracles`:

```console
python gen_firstorder3d_pyradiomics.py
```

PyRadiomics computes first-order features from the original intensities. Only `Entropy` and
`Uniformity` use the discretized histogram, so `binCount=20` affects those two features and matches
Nyxus' `GREYDEPTH=-20` setting.

## Golden reproduction

Both recorded dependency stacks reproduce all 17 checked-in goldens exactly: relative error is zero
for every value in `tests/test_3d_firstorder_pyradiomics.h`. The generator prints the versions it
actually runs and verifies that every mapped feature has a header pin. PyRadiomics `TotalEnergy`
remains deliberately unmapped because Nyxus has no corresponding feature in this table.

The generator performs these mechanical checks over the pin set:

| check | value |
|---|---:|
| `3UNIFORMITY` in [0,1] | 0.1003710937 |
| `3ENTROPY >= 0` | 3.593829137 |
| `3RANGE == 3MAX - 3MIN` | 441 |
| `3MIN <= 3P10 <= 3MEDIAN <= 3P90 <= 3MAX` | 442 |
| `3ROOT_MEAN_SQUARED^2 == 3MEAN^2 + 3VARIANCE` | 448.4583107 |
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION <= 3MEAN_ABSOLUTE_DEVIATION` | 33.49484999 |

## Nyxus agreement and tolerance bands

The previous assertions all used `agrees_gt(..., 10)`, a 10% relative band. Measuring each Nyxus
result against its reproduced PyRadiomics golden supports three narrower bands:

| band | features | worst measured residual |
|---|---|---:|
| `rel=1e-9` | 12 same-definition features | less than 1e-12 |
| `rel=1e-2` | `3P10`, `3P90`, `3INTERQUARTILE_RANGE`, `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | 4.99e-3 |
| `rel=1e-3` | `3VARIANCE` | 2.08e-4 |

The four percentile-derived features use Nyxus' 100-bin interpolated CDF, while PyRadiomics uses
sample percentiles. Their values agree within `rel=1e-2`; the largest measured residual is on
`3ROBUST_MEAN_ABSOLUTE_DEVIATION`.

For `3VARIANCE`, PyRadiomics reports population variance (N) and Nyxus reports sample variance
(N-1). On this ROI the resulting 2.08e-4 residual agrees within `rel=1e-3`. Nyxus'
`3VARIANCE_BIASED` is the term-for-term population-variance counterpart, but it is not part of this
existing PyRadiomics table.

Measured non-exact comparisons:

| feature | Nyxus | PyRadiomics | relative residual |
|---|---:|---:|---:|
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | 33.3279 | 33.4948499916 | 4.99e-3 |
| `3P10` | 362.675 | 362 | 1.86e-3 |
| `3P90` | 527.438 | 527 | 8.30e-4 |
| `3INTERQUARTILE_RANGE` | 78.9528 | 79 | 5.98e-4 |
| `3VARIANCE` | 4197.79 | 4196.911126692708 | 2.08e-4 |

The assertion helper keeps its two-argument interface because the generic 3D coverage sweep calls
it with the feature enum and name. Its enum-based switch selects the measured band without
classifying features by string.

## Provenance correction

The former header comment described `ut_inten.nii`, label 57, and GLCM settings even though the
test loads the COMPAT MRI/liver fixture at label 1. The comment now records the fixture, label,
PyRadiomics version, recipe, and checked-in generator actually used. The unused `raw_nifti.h`
include was also removed.
