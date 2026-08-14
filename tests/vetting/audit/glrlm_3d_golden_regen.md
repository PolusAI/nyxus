# Regenerating the 3D GLRLM goldens

Two benchmarks. They share the same grey binning but sit on different fixtures, so they are **not**
comparable to each other.

## PyRadiomics goldens — `test_3d_glrlm_pyradiomics.h`

Recipe `glrlm3d.pyradiomics_bincount20`, on the compat phantom
(`tests/data/nifti/compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1).

```
python tests/vetting/oracles/gen_glrlm3d_pyradiomics.py
```

The generator prints a paste-ready table and re-verifies every pinned golden, exiting non-zero on any
mismatch, any pin it cannot produce, any golden the oracle leaves unpinned, and any bounded feature
the oracle itself reports out of range. It needs PyRadiomics 3.0.1, i.e. Python ≤ 3.9 — a conda env,
not the build env. Equivalent CLI form:

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
  glrlm:
```

The Nyxus side must match it with `GREYDEPTH=100`, `IBSI=false`, `GLRLM_GREYDEPTH=-20` — negative
activates radiomics binCount-based binning, so the magnitude is the bin count. Setting only the
generic `GREYDEPTH` does nothing here: `D3_GLRLM_feature` bins on `GLRLM_GREYDEPTH` alone, and left
unset that defaults to 0, i.e. no binning, with the features running on raw intensities.

**The single most important convention.** PyRadiomics reports **one value per feature over its whole
direction set**. That is the Nyxus `*_AVE` aggregation over the 13 3D angles — *not* a per-angle
value. A golden from this run is therefore the reference for both the per-angle base feature (through
`calc_ave`) and the stored `*_AVE` feature. Do not pin it against a single angle.

**Name mapping** — one-to-one, and unlike GLCM no pair of them collides:

| Nyxus | PyRadiomics | | Nyxus | PyRadiomics |
|---|---|---|---|---|
| `3GLRLM_SRE` | `ShortRunEmphasis` | | `3GLRLM_RP` | `RunPercentage` |
| `3GLRLM_LRE` | `LongRunEmphasis` | | `3GLRLM_GLV` | `GrayLevelVariance` |
| `3GLRLM_GLN` | `GrayLevelNonUniformity` | | `3GLRLM_RV` | `RunVariance` |
| `3GLRLM_GLNN` | `GrayLevelNonUniformityNormalized` | | `3GLRLM_RE` | `RunEntropy` |
| `3GLRLM_RLN` | `RunLengthNonUniformity` | | `3GLRLM_LGLRE` | `LowGrayLevelRunEmphasis` |
| `3GLRLM_RLNN` | `RunLengthNonUniformityNormalized` | | `3GLRLM_HGLRE` | `HighGrayLevelRunEmphasis` |
| `3GLRLM_SRLGLE` | `ShortRunLowGrayLevelEmphasis` | | `3GLRLM_LRLGLE` | `LongRunLowGrayLevelEmphasis` |
| `3GLRLM_SRHGLE` | `ShortRunHighGrayLevelEmphasis` | | `3GLRLM_LRHGLE` | `LongRunHighGrayLevelEmphasis` |

All 16 have a counterpart, so nothing in this family is vetted through an identity the way six GLCM
features are.

**Tolerance.** Nyxus lands on PyRadiomics to double precision on 15 of the 16, so they are asserted
at `rel=1e-9`. `3GLRLM_RE` is the exception at `rel=5e-3`: it is the family's only sum over
logarithms and Nyxus evaluates it through `fast_log10` with an `EPSILON` guard, measured 3.9e-4 away.
The 2D family carries the same exception for the same reason.

## Regression drift guards — `test_3d_glrlm_regression.h`

Recipe `glrlm3d.regression_ut_phantom`: the segmented phantom (`phantoms/ut_inten.nii` +
`phantoms/ut_mask57.nii`, label 57) at `GLRLM_GREYDEPTH=-20`, averaged over the 13 angles. No oracle —
Nyxus' own values.

```
runAllTests --gtest_filter=*3D_GLRLM_DUMP_REGRESSION*
```

`test_3d_glrlm_dump_regression()` prints the whole table at 17 significant digits in the shape
`glrlm_3d_regression_ref_vals` wants; paste it over the table. It runs the same settings the
assertions use, so the two cannot drift apart.

Sanity checks worth running on any regenerated set, both of which the pre-2026-08 goldens failed:

- `SRE`, `RP`, `GLNN` and `RLNN` are bounded in **[0,1]** and `LRE` is **≥ 1**. A value outside those
  is a broken golden — or, for `RP` at positive grey depths, the open implementation defect recorded
  in `glrlm_3d_pyradiomics_vetting_report.md`. That is why this benchmark uses binCount binning and
  not a positive `GLRLM_GREYDEPTH`.
- The magnitudes must be consistent with the bin count. Unbinned, `HGLRE` reaches 4.3e6; at 64 grey
  levels the ceiling is ~4e3. An `HGLRE` in the millions means `GLRLM_GREYDEPTH` was never set.

## Adding a 3D header that needs the segmented phantom

`get_3d_segmented_phantom()` is **defined once**, in `test_3d_glcm_pyradiomics.h`. Every other 3D
header forward-declares it:

```cpp
static std::tuple<std::string, std::string, int> get_3d_segmented_phantom();
```

Carrying a second definition is a redefinition error in the single `test_all.cc` translation unit,
which is what kept this file — and still keeps `test_3d_{gldm,glszm,ngtdm}_regression.h` and
`test_3d_firstorder_matlab.h` — out of the build.

## Coverage artifact

```
python tests/vetting/audit/scan_glrlm3d_coverage.py           # rewrite
python tests/vetting/audit/scan_glrlm3d_coverage.py --check   # drift + acceptance check
```

The scanner reads the feature→test mapping out of the test sources rather than a hand-written list,
and `--check` also asserts that every `vetted` row is backed by an oracle test naming the oracle the
row names. Two 3D-specific wrinkles it handles: tests name features by enum
(`Feature3D::GLRLM_SRE_AVE`, including through the `using F = Nyxus::Feature3D` alias) while the
registry carries the leading dimension digit (`3GLRLM_SRE_AVE`); and `test_3d_glrlm_coverage.h`
instantiates parameterized suites over the family's featureset, so what it touches is decided at
runtime and it is credited to the whole family rather than scanned.
