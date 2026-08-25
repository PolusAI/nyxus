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

Recipes: `glcm.ibsi_identity`, `ih.ibsi_fbn`, `gldm.ibsi_phantom_2d`, `gldzm.ibsi_phantom_2d`,
`glszm.ibsi_phantom_2d`, `ngldm.ibsi_phantom_2d`, `ngtdm.ibsi_phantom_2d`, `ngtdm.default_fbn100`.

Tests reaching it today: `test_2d_{firstorder,glcm,gldm,gldzm,glszm,ngldm,ngtdm}_ibsi.h`,
`test_2d_gldm_pyradiomics.h`, `test_2d_{gldzm,glszm,ngldm,ngtdm}_mirp.h`,
`test_2d_{glcm,gldzm,glrlm,glszm,ngtdm}_regression.h`, `test_2d_glrlm_common.h`,
`test_2d_{gldm,gldzm,glszm,intensity_histogram}_common.h`.

Aggregation matters here: the IBSI 2D-averaged values are the mean over the four slices featurised
one at a time, which is what the per-family fixtures do.

**One recipe on this fixture is not an IBSI one.** `ngtdm.default_fbn100` runs the same four slices
through Nyxus' default mode, where the intensities are re-binned to a fixed grey count instead of
using the phantom's own 1..6. The fixture is shared; the quantity is not, and no `ibsi` assertion can
be made at that recipe.

**Both aggregations are worth pinning, not just the published one.** A four-slice mean is blind to
errors in two slices that cancel: perturbing `GLDM_LDE_z1` by +0.001 and `GLDM_LDE_z3` by -0.001
leaves the mean unmoved to its last digit. Where a tool exposes the per-slice values — PyRadiomics
does, the published IBSI consensus does not — the per-slice table is the stronger assertion and the
mean is the compatibility one. `test_2d_gldm_pyradiomics.h` pins both, 14 means and 56 slice
values; `test_2d_gldzm_mirp.h` and `test_2d_glszm_mirp.h` each pin 16 means and 64 slice values
against mirp, which also exposes per-slice output, and `test_2d_ngtdm_mirp.h` pins 5 means and 20
slice values the same way.

---

## `bench_dsb2018_2d` — four DSB2018 nuclei ROIs

| | |
|---|---|
| Data | `dsb_data` in `tests/test_dsb2018_data.h`, loaded by `load_test_roi_data()` in `test_main_nyxus.h` |
| ROI | the whole image is the ROI in each case — no mask, no background inside the bounding box |
| Shape | four 8-bit crops, 10×9, 16×14, 11×17 and 13×15 pixels; nuclei from the 2018 Data Science Bowl set |
| Why it exists | real-image intensity texture at a size a unit test can carry as a literal, on four ROIs with different aspect ratios and different amounts of zero border — which is what makes it a spread rather than one shape |

Recipes: `gabor.cpp_static_defaults`, `gabor.python_raw_defaults`.

Tests reaching it today: `test_2d_gabor_skimage.cc`, `test_2d_gabor_mechanics.h`.

**These ROIs are small relative to a Gabor kernel, and that is load-bearing rather than incidental.**
At `gamma = 0.1` the analytic filter runs to hundreds of pixels a side while these ROIs are 9–17
pixels, so Nyxus' 16×16 crop decides most of the value. A benchmark swap here does not preserve the
goldens' meaning — see `audit/gabor_2d_skimage_vetting_report.md` §4.1.

---

## `bench_scene7_5roi_enclosed` — five labelled ROIs in one 7×7 scene

| | |
|---|---|
| Data | `neighborhood2d_scene_labels` in `tests/test_data.h`, built into a `roiData` map by `calculate_neighbor_feature_values()` in `tests/test_2d_neighbor_common.h` |
| ROI | five of them — labels 1–5, 27 pixels total, all in one scene |
| Shape | a 7×7 footprint spanning x 2–8, y 2–8: label 1 is the 3×3 centre block, labels 2–5 are 2×2 and 2×3 blocks placed left, above, right and below it |
| Why it exists | the neighbour graph is a property of a *scene*, not of a single ROI, so this is the smallest fixture on which every neighbour quantity is non-degenerate — one ROI with four neighbours and four with exactly one each |

Recipes: `neighbor.scene2d_radius1`.

Tests reaching it today: `test_2d_neighbor_{cellprofiler,analytic,regression,invariant}.h`,
`tests/python/test_2d_neighbor_invariant.py`.

**Label 1 being fully enclosed is the load-bearing property, not a layout accident.** It is the only
ROI whose every contour pixel is 8-adjacent to some neighbour, which is what makes
`PERCENT_TOUCHING = 100` a closed form rather than a snapshot, and it is the only ROI with two
in-radius neighbours — so it is also the only one where `CLOSEST_NEIGHBOR2_*` is anything but a
structural zero. Shrinking the scene or moving one block off label 1 costs both assertions.
