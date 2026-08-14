# Regenerating the 3D GLCM goldens

Two benchmarks with different binning, so they are **not** comparable to each other.

## PyRadiomics goldens — `test_3d_glcm_pyradiomics.h`

Recipe `glcm3d.pyradiomics_bincount20`, on the compat phantom
(`tests/data/nifti/compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`).

```
pyradiomics <intensity>.nii <mask>.nii --param settings1.yaml
```

with `settings1.yaml`:

```yaml
setting:
  binCount: 20
  label: 1
  interpolator: 'sitkBSpline'
  resampledPixelSpacing:
  weightingNorm:
imageType:
  Original: {}
featureClass:
  glcm:
    - 'Autocorrelation'
    - 'JointAverage'
    ...
```

The Nyxus side must match it with `GREYDEPTH=100`, `IBSI=false`, `GLCM_GREYDEPTH=-20` (negative
activates radiomics binCount-based binning, so the magnitude is the bin count), `GLCM_OFFSET=1`,
`GLCM_SPARSEINTENS=true`.

**The single most important convention.** PyRadiomics reports **one value per feature over its whole
direction set**. That is the Nyxus `*_AVE` aggregation over the 13 3D angles — *not* a per-angle
value. A golden from this run is therefore the reference for both the per-angle base feature (through
`calc_ave`) and the stored `*_AVE` feature. Do not pin it against a single angle.

**Name mapping** — PyRadiomics names differ from the Nyxus ones, and three collide confusingly:

| Nyxus | PyRadiomics | | Nyxus | PyRadiomics |
|---|---|---|---|---|
| `3GLCM_ASM` | `JointEnergy` | | `3GLCM_JVAR` | `SumSquares` |
| `3GLCM_JE` | `JointEntropy` | | `3GLCM_JMAX` | `MaximumProbability` |
| `3GLCM_ACOR` | `Autocorrelation` | | `3GLCM_INFOMEAS1` | `Imc1` |
| `3GLCM_JAVE` | `JointAverage` | | `3GLCM_INFOMEAS2` | `Imc2` |
| `3GLCM_IV` | `InverseVariance` | | `3GLCM_DIFAVE` | `DifferenceAverage` |

Note `3GLCM_ASM` maps to *JointEnergy* and `3GLCM_JVAR` to *SumSquares* — neither is the
similarly-named PyRadiomics feature. PyRadiomics `MCC` has no Nyxus counterpart.

**Six features have no PyRadiomics golden of their own** and must not be given one: `DIS`, `ENERGY`,
`ENTROPY`, `HOM1`, `SUMVARIANCE`, `VARIANCE`. PyRadiomics deprecates `DIS` as equivalent to
`DifferenceAverage` and does not report the others under their own name. They are vetted through the
identities `DIS≡DIFAVE`, `ENERGY≡ASM`, `ENTROPY≡JE`, `HOM1≡ID`, `SUMVARIANCE≡CLUTEND`,
`VARIANCE≡JVAR`, asserted at 1e-6 by `test_3d_glcm_{equivalence_dump,ave_equivalence}_pyradiomics`.

## Regression drift guards — `test_3d_glcm_regression.h`

Recipe `glcm3d.regression_ut_phantom`: the segmented phantom (`phantoms/ut_inten.nii` +
`phantoms/ut_mask57.nii`, label 57) at **binCount 100** (`GLCM_GREYDEPTH=-100`), averaged over the 13
angles. No oracle — Nyxus' own values.

```
runAllTests --gtest_filter=*3D_GLCM_DUMP_REGRESSION*
```

`test_3d_glcm_dump_regression()` prints the whole table at 17 significant digits in the shape
`glcm_3d_regression_ref_vals` wants; paste it over the table. It runs the same settings the
assertions use, so the two cannot drift apart.

Sanity checks worth running on any regenerated set, both of which the pre-2026-08 goldens failed:

- `ID`, `IDM`, `IDN`, `IDMN`, `JMAX` and `ASM` are bounded in **[0,1]**, and `CORRELATION` in
  **[-1,1]**. A value outside those is a broken golden, not a surprising measurement.
- `SUMVARIANCE == CLUTEND` and `DIS == DIFAVE` exactly (to ~1e-15). These identities hold by
  construction; if a regenerated table breaks one, the run was misconfigured.

## Adding a 3D header that needs the segmented phantom

`get_3d_segmented_phantom()` is **defined once**, in `test_3d_glcm_pyradiomics.h`. Every other 3D
header forward-declares it:

```cpp
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();
```

Carrying a second definition is a redefinition error in the single `test_all.cc` translation unit,
which is what kept `test_3d_glcm_regression.h` — and still keeps
`test_3d_{gldm,glrlm,glszm,ngtdm}_regression.h` and `test_3d_firstorder_matlab.h` — out of the build.

## Coverage artifact

```
python tests/vetting/audit/scan_glcm3d_coverage.py           # rewrite
python tests/vetting/audit/scan_glcm3d_coverage.py --check   # drift + acceptance check
```

Two 3D-specific wrinkles the scanner handles, both worth copying for the remaining 3D families:
tests name features by enum (`Feature3D::GLCM_ACOR_AVE`) while the registry carries the leading
dimension digit (`3GLCM_ACOR_AVE`), including through the `using F = Nyxus::Feature3D` alias; and
`test_3d_glcm_coverage.h` instantiates parameterized suites over the family's featureset, so what it
touches is decided at runtime and it is credited to the whole family rather than scanned.
