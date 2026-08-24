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

`test_3d_glrlm_dump_regression()` prints **both** of the file's tables at 17 significant digits, each
in the shape it wants: the `[3DGLRLM-REGEN]` block is `glrlm_3d_regression_ref_vals`, the
`[3DGLRLM-REGEN-GREY64]` block is `glrlm_3d_regression_grey64_ref_vals`. Paste each over the table
above it. Both run through `run_3d_glrlm_pipeline()`, the same helper the assertions use, at their
own profile, so a table and its assertions cannot drift apart.

Sanity checks worth running on any regenerated set, both of which the pre-2026-08 goldens failed:

- `SRE`, `RP`, `GLNN` and `RLNN` are bounded in **[0,1]** and `LRE` is **≥ 1**. A value outside those
  is a broken golden — or, for `RP` at positive grey depths, the open implementation defect recorded
  in `glrlm_3d_pyradiomics_vetting_report.md`. That is why the `glrlm3d.regression_ut_phantom`
  benchmark uses binCount binning and not a positive `GLRLM_GREYDEPTH`. The grey64 table below is the
  one deliberate exception and it does trip this check — 4 of `RP`'s 13 angles read 1.0231 — which is
  the filed defect showing through, not a broken golden.
- The magnitudes must be consistent with the bin count. Unbinned, `HGLRE` reaches 4.3e6; at 64 grey
  levels the ceiling is ~4e3. An `HGLRE` in the millions means `GLRLM_GREYDEPTH` was never set.

## grey64 table and the retired sweep — `glrlm_3d_regression_grey64_ref_vals`

Recipe `glrlm3d.regression_ut_phantom_grey64`: the same segmented phantom at `GLRLM_GREYDEPTH=+64`,
which `bin_intensities_3d()` routes through `matlab_grey_binning` — a fixed 64-level count — instead
of the binCount binning derived from the ROI's own min/max.

It pins **every slot of all 32 features**: the 13 angled values of each base feature and the single
mean its `*_AVE` twin stores. The family's retired coverage sweep pinned only the 16 means, and
`revet.txt` §9's first rule is why that is not enough — two per-angle errors of opposite sign leave
the mean unmoved. Negative control, recorded here because §9 asks for the exact perturbation: adding
`+8.458e-08` to `3GLRLM_SRE` angle 0 and subtracting it from angle 1 leaves
`TEST_3D_GLRLM_SRE_AVE_GREY64_REGRESSION` **passing** while
`TEST_3D_GLRLM_SRE_GREY64_REGRESSION` **fails and names both elements**.

The 16 means are the ones `test_3d_glrlm_coverage.h` carried as
`glrlm_3d_regression_coverage_ref_vals`, asserted through the parameterized
`GLRLM_UNVETTED_LOCAL_REGRESSION` suite, which reported a failure as a sanitized test-param string
rather than a feature name. They agree with the retired literals to `rel<=3.9e-16`, except
`3GLRLM_RE_AVE` at `4.1e-10` — the family's only sum over logarithms, evaluated through the
float-precision `Nyxus::fast_log10`. That split is what sets the bands: `rel=1e-9` for everything
else, `rel=1e-6` for `RE` and `RE_AVE`, which leaves `RE` a factor of ~2400 of headroom where `1e-9`
would leave 2.4.

The recipes had to be proved equal rather than assumed. `D3_GLRLM_feature::calculate()` reads exactly
three settings — `SOFTNAN`, `IBSI`, `GLRLM_GREYDEPTH` — and `make_3d_coverage_settings()` and
`run_3d_glrlm_pipeline()` agree on all three (`0.0`, `false`, `64`) over the same phantom and label,
which is why the aggregates could move unchanged.

Checks run mechanically over the whole table rather than spot-checked (`revet.txt` §9):

| check | result |
|---|---|
| every `*_AVE` pin == arithmetic mean of its own 13 per-angle pins | worst `rel=2.9e-16` (`LRLGLE_AVE`), passes at `rel=1e-12` |
| `SRE`, `GLNN`, `RLNN` ∈ [0,1] over all 13 angles and the mean | pass |
| `RP` ∈ [0,1] | **fails on 4 of 13 angles** (1.0231) — the filed defect, see below |
| `LRE` ≥ 1 | pass |
| `RE`, `GLV`, `RV` ≥ 0 | pass |

The regenerator walks `UserFacing_3D_featureNames` rather than the table's own keys, so a feature the
run produces that the table pins nothing for prints anyway — the reverse check §9 asks for.

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
registry carries the leading dimension digit (`3GLRLM_SRE_AVE`); and a test function's *suffix*
decides what kind of coverage it contributes, so only a `_pyradiomics` function can put an oracle
token on a row while the `_regression` ones fill the `Reg_Test_Name` column.
