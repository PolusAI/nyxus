# Benchmark registry

The fixtures assertions run on, one entry each: what it is, why it exists, and what uses it
(SPEC §6.3). A `benchmark` id in `oracle_coverage.csv` must be defined here — a row naming a
fixture nobody can look up records no more than a row naming none.

Ids follow §6.3: `bench_<shape><size>_<property>` for inline fixtures, and the canonical name for
standard phantoms.

Seeded with the fixtures the 3D NGLDM rows point at, plus the two other phantoms already carrying
their own recipes, so the file is a registry rather than a single entry. Rows whose `benchmark`
column is still empty are the backlog; each family fills its own as it is revisited. The "used by"
lists are what this tree shows today — the recipe ids are the stable half.

---

## `bench_ut57_3d` — the segmented `ut_` phantom

| | |
|---|---|
| Files | `tests/data/nifti/phantoms/ut_inten.nii` + `tests/data/nifti/phantoms/ut_mask57.nii` |
| ROI | label **57** |
| Shape | 100×100×100 voxels at 1×1×1 spacing; the ROI is **274 432** voxels, ~27% of the volume |
| Why it exists | a solid single-label 3D ROI, large enough that a 13-direction texture matrix is well populated in every direction and small enough to featurise inside a unit test |

Recipes: `ngldm3d.regression_ut_phantom`, `ngldm3d.mirp_fbn64`.

Tests reaching it today: `test_3d_ngldm_regression.h`, `test_3d_coverage_common.h`,
`test_3d_{glcm,gldm,gldzm}_regression.h`, `test_3d_morphology_matlab.h`, and the
`get_3d_segmented_phantom()` helper the 3D `_pyradiomics` files define (several of those assert on
`bench_compat_liver_3d` instead — the helper being present is not the same as the assertion using
it).

**Not interchangeable with `bench_compat_liver_3d`.** The two differ in ROI size and in the bin
counts their recipes use, so a golden pinned on one says nothing about the other. An earlier 3D GLCM
provenance comment named the wrong one of these two, which is the mistake this entry exists to
prevent.

---

## `bench_compat_liver_3d` — the COMPAT phantom, liver label

| | |
|---|---|
| Files | `tests/data/nifti/compat_int/compat_int_mri.nii` + `tests/data/nifti/compat_seg/compat_seg_liver.nii` |
| ROI | label **1** |
| Why it exists | an MRI-like intensity distribution over an anatomical segmentation — the fixture the PyRadiomics comparisons are run on, whose grey-level spread suits `binCount` binning |

Tests reaching it today: `test_3d_glcm_pyradiomics.h`, `test_3d_firstorder_pyradiomics.h`.

---

## `ibsi_digital_phantom` — the IBSI reference phantom, 4 slices

| | |
|---|---|
| Data | `ibsi_phantom_z1..z4_{intensity,mask}` in `tests/test_data.h` |
| Shape | four 2D slices; grey levels are already discrete 1..6, so no binning is applied |
| Why it exists | the fixture the published IBSI consensus values are defined on, so it is the only benchmark on which an `ibsi` assertion means anything |

Recipes: `glcm.ibsi_identity`, `ih.ibsi_fbn`.

Tests reaching it today: `test_2d_{firstorder,glcm,gldm,gldzm,glszm,ngldm,ngtdm}_ibsi.h`,
`test_2d_{glcm,glrlm,glszm,ngtdm}_regression.h`, `test_2d_glrlm_common.h`,
`test_2d_intensity_histogram_common.h`.

Aggregation matters here: the IBSI 2D-averaged values are the mean over the four slices featurised
one at a time, which is what the per-family fixtures do.
