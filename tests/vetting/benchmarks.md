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

## `bench_irregular13x18_intensity` — the canonical 2D first-order ROI

| | |
|---|---|
| Fixture | `pixelIntensityFeaturesTestData` in `tests/test_data.h` |
| ROI | 154 foreground pixels in a 13x18 bounding box |
| Intensity domain | integer values 11079..64090; sum 5015224 |
| Why it exists | an irregular, nonuniform ROI that exercises extrema, moments, percentiles and slide-relative range without image-loader transformations |

Recipes: `firstorder2d.matlab_native`, `firstorder.pyradiomics_default`,
`firstorder2d.regression_default`, and `firstorder2d.regression_greydepth20`.
Tests: `test_2d_firstorder_{matlab,pyradiomics,regression}.h`.

---

## `bench_ut57_3d` — the segmented `ut_` phantom

| | |
|---|---|
| Files | `tests/data/nifti/phantoms/ut_inten.nii` + `tests/data/nifti/phantoms/ut_mask57.nii` |
| ROI | label **57** |
| Shape | 100×100×100 voxels at 1×1×1 spacing; the ROI is **274 432** voxels, ~27% of the volume |
| Why it exists | a solid single-label 3D ROI, large enough that a 13-direction texture matrix is well populated in every direction and small enough to featurise inside a unit test |

Recipes: `firstorder3d.matlab_native`, `firstorder3d.regression_ut_phantom`,
`ngldm3d.regression_ut_phantom`, `ngldm3d.mirp_fbn64`, `gldzm3d.regression_ut_phantom`,
`gldzm3d.mirp_fbn64`, `ngtdm3d.regression_ut_phantom`, `glszm3d.regression_ut_phantom`,
`gldm3d.regression_ut_phantom`, `gldm3d.regression_ut_phantom_nobinning`.
Tests reaching it today: `test_3d_ngldm_regression.h`, `test_3d_coverage_common.h`,
`test_3d_{glcm,gldm,gldzm,glszm,ngtdm}_regression.h`, `test_3d_firstorder_{matlab,regression}.h`,
`test_3d_morphology_matlab.h`, and the
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

Recipes: `glszm3d.pyradiomics_bincount20`.

Tests reaching it today: `test_3d_glcm_pyradiomics.h`, `test_3d_firstorder_pyradiomics.h`,
`test_3d_glszm_pyradiomics.h`, `test_3d_glszm_mechanics.h`, `test_3d_gldm_pyradiomics.h`,
`tests/python/test_nyxus.py::test_3d_glszm_compatibility`,
`tests/python/test_nyxus.py::test_3d_gldm_compatibility`.

**It is not the fixture for the family's connectivity or its IBSI branch.** Its size-zone matrix has
186 populated cells, which nobody checks by hand, and its grey levels after `binCount` binning are
contiguous 1..20, which makes the IBSI row index and the one beside it the same number. Those two
questions are asked on `bench_cube4x4x3_zcross` and `bench_cube3_gapped_levels` below.

**Its liver segmentation has background, so PyRadiomics' public extractor loads it.** Worth stating
because that is not automatic: `imageoperations.getMask()` raises "No labels found in this mask"
whenever a mask has a single unique value, and a 3D fixture whose label covers every voxel has to be
reached by constructing the feature class directly. This one does not.

**The ROI is 4 800 voxels in a 30x87x4 bounding box** at 0.677x0.677x4.8 mm spacing, and its
intensities span 212..653. Its largest single-grey-level connected component is 634 voxels, which is
why the 3D GLSZM size-zone matrix is 20x634 and mostly empty.

The 3D GLDM assertions reach past the fourteen scalars to the **dependence matrix** on this ROI: 163
non-empty cells over `Ng=20` grey levels and `Nd=15` dependences, with `Nz = Np = 4800` because
PyRadiomics allows incomplete zones and so gives every ROI voxel exactly one dependence zone. Those
numbers are properties of this benchmark, not of the family, and any other family intercepting a
matrix here should reproduce the same 4800.

Recipes: `glcm3d.pyradiomics_bincount20`, `firstorder3d.pyradiomics_bincount20`,
`gldm3d.pyradiomics_bincount20`.

---

## `bench_compat_ngtdm_3d` — the 4x4x3 NGTDM phantom

| | |
|---|---|
| Files | `tests/data/nifti/compat_int/compat_int_ngtdm_3d.nii` + `tests/data/nifti/compat_seg/compat_seg_ngtdm_3d.nii` |
| ROI | label **57**, which is every voxel |
| Shape | 4x4x3 at 1x1x1 spacing: one populated 4x4 slice of the discrete levels 1..5 between two all-zero slices, 48 voxels in total |
| Why it exists | six grey levels over 48 voxels, small enough that the whole NGTDM is 18 numbers a reader can check by hand, and constructed so that PyRadiomics' `binWidth=1` discretisation and Nyxus' zero-min correction land on the same level set |

Recipes: `ngtdm3d.pyradiomics_binwidth1`, `ngtdm3d.pyradiomics_binwidth1_r2`. The two differ only
in `NGTDM_RADIUS` (1 and 2). The fixture supports no third: at radius 3 a Chebyshev neighbourhood
already spans the whole 4×4×3 volume from every voxel, so radius 3 and radius 4 compute one matrix.

Tests reaching it today: `test_3d_ngtdm_pyradiomics.h`, `test_3d_ngtdm_mechanics.h`,
`tests/python/test_nyxus.py::test_3d_ngtdm_compatibility`.

**Its mask has no background, and that has a consequence.** All 48 voxels carry label 57, so
PyRadiomics' `imageoperations.getMask()` rejects it outright — `numpy.unique` on the mask has one
entry and it raises "No labels found in this mask (i.e. nothing is segmented)!". Any oracle run
against this fixture has to construct the feature class directly rather than go through
`RadiomicsFeatureExtractor`; `oracles/gen_ngtdm3d_pyradiomics.py` does.

The all-zero slices are ROI voxels, not padding. They become grey level 1 on both sides and are two
thirds of the volume, which is why `n_1 = 32` dominates the matrix.

---

## `ibsi_digital_phantom` — the IBSI reference phantom, 4 slices

| | |
|---|---|
| Data | `ibsi_phantom_z1..z4_{intensity,mask}` in `tests/test_data.h` |
| Shape | four 2D slices; grey levels are already discrete 1..6, so no binning is applied |
| Why it exists | the fixture the published IBSI consensus values are defined on, so it is the only benchmark on which an `ibsi` assertion means anything |

Recipes: `firstorder2d.ibsi_digital_phantom`, `glcm.ibsi_identity`, `ih.ibsi_fbn`,
`gldm.ibsi_phantom_2d`, `gldzm.ibsi_phantom_2d`,
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

## `bench_cube4x4x3_zcross` — a 4x4x3 cube whose zones cross the slices

| | |
|---|---|
| Data | `glszm_3d_zcross_volume` in `tests/test_3d_glszm_common.h`, run through `run_3d_glszm_on_volume()` |
| ROI | every non-zero voxel — 17 of the 48, over grey levels 1..4 |
| Shape | three populated 4x4 slices; nine 26-connected zones of sizes 1..3 |
| Why it exists | the family's connectivity check, on a volume whose every zone can be counted by eye — which the phantom's 186-cell matrix cannot be |

Recipes: none of its own — it is asserted under `glszm3d.pyradiomics_ibsi_gapped`'s sibling case
`TEST_3D_GLSZM_SMALLMATRIX_PYRADIOMICS`, at the family's no-binning setting.

Tests reaching it today: `test_3d_glszm_pyradiomics.h`.

**Every one of its nine zones is there to separate one neighbourhood from another**, which is the
property that makes it a fixture rather than a small input: a vertical run (dz=±1, dy=dx=0), an
in-slice diagonal, a z-edge join, a z-corner join no 18-neighbourhood makes, and two same-level
voxels two slices apart that must stay two zones. Counting under the readings this could be confused
with gives 9 zones at 26-connectivity, 10 at 18-, 13 at 6-, and 13 for a purely 2D 8-neighbour pass.

**Its predecessor had one populated slice between two empty ones**, so every `dz != 0` neighbour of
every voxel was background: 26-, 18- and 2D 8-connectivity all produced the same nine zones there,
and only 6-connectivity differed. The fixture was measuring in-slice connectivity under a 3D name.

---

## `bench_cube3_gapped_levels` — a 3x3x3 cube with grey levels 1, 3 and 5

| | |
|---|---|
| Data | `glszm_3d_gapped_volume` in `tests/test_3d_glszm_common.h`, run through `run_3d_glszm_on_volume()` |
| ROI | every non-zero voxel — 9 of the 27 |
| Shape | three populated 3x3 slices; six 26-connected zones, sizes 1..3, over three occupied grey levels |
| Why it exists | the only fixture on which the `IBSI=true` branch of `D3_GLSZM_feature::calculate()` is distinguishable from the code beside it |

Recipes: `glszm3d.pyradiomics_ibsi_gapped`.

Tests reaching it today: `test_3d_glszm_pyradiomics.h`, `test_3d_glszm_mechanics.h`.

**The gap is the whole point.** `calculate()` indexes a zone's row as `zone.first - 1` when IBSI is
on and as the position of that level in `I` when it is off, and reports `Ng` as `max(I)` against
`I.size()`. On any fixture whose occupied levels run contiguously from 1 those are the same numbers.
Here they are not, and both sides still agree: PyRadiomics reports `Ng = 5` for three occupied
levels, leaving rows 2 and 4 empty, and so does Nyxus.

**It also carries the family's connectivity questions in miniature** — a vertical pair, a level-3
zone assembled from two z-corner steps, and three level-5 voxels that are pairwise non-adjacent
(`dx = 2` in-slice, `dz = 2` across slices) and must stay three zones.

---

## `bench_cube2_constant` — a 2x2x2 cube of one intensity

| | |
|---|---|
| Data | `glszm_3d_constant_volume` in `tests/test_3d_glszm_common.h`, run through `run_3d_glszm_on_volume()` |
| ROI | all 8 voxels, every one of them intensity 7 |
| Shape | one grey level, one 26-connected zone of size 8, a size-zone matrix with a single populated cell |
| Why it exists | the smallest ROI that reaches `calculate()`'s `aux_min == aux_max` intercept, which is the one input on which this family answers nothing |

Recipes: `glszm3d.regression_constant_roi`.

Tests reaching it today: `test_3d_glszm_regression.h`.

**It is a fixture for a divergence, not for a value.** A constant-intensity ROI is fully populated
and has sixteen finite features over it — on this cube `SAE = 1/64`, `LAE = 64`, `ZE = 0`, `GLV = 0`,
`ZP = 1/8` — and Nyxus returns the soft-NaN sentinel for all sixteen instead. The assertions pin the
sentinel, so they will fail the day the intercept is narrowed, which is the point:
`tests/vetting/matrix/glszm3d.md` dispositions the cell `VALID-prod-only` rather than `INVALID`, and
records that what it pins is a known defect.

**Its intensity is 7 and its sentinel is `-98765`, both deliberately.** A zone landing on the wrong
row is a visibly wrong level at 7, and a sentinel of `-98765` cannot be satisfied by the zero-filled
buffer `initialize_fvals()` hands `calculate()` — a default `0.0` sentinel would let a feature that
was never written pass.

**Not interchangeable with the family's other small cubes.** `bench_cube4x4x3_zcross` and
`bench_cube3_gapped_levels` both carry several grey levels, so neither reaches the intercept at all.

---

## `bench_constant_roi_3d` — a nonempty ROI of one intensity

| | |
|---|---|
| Data | built inline in `test_3d_gldm_constant_roi_regression()` (`tests/test_3d_gldm_regression.h`) |
| ROI | one label filling the whole cube |
| Shape | 4×4×3 voxels, every voxel intensity **7**, so `aux_min == aux_max` |
| Why it exists | the degenerate ROI a texture family has to have an answer for. It is the smallest fixture on which `aux_min == aux_max` holds while the ROI is *not* blank, which is the case Nyxus and PyRadiomics answer differently |

Recipes: `gldm3d.regression_constant_roi`.

Tests reaching it today: `test_3d_gldm_regression.h`.

**Reachable in production, and the fixture is deliberately tiny.** Any segmentation over a flat
region produces this shape, and 4×4×3 is small enough that PyRadiomics' whole dependence matrix on
the same voxels fits in the recipe entry — `[8, 20, 16, 4]` over one grey level — so the divergence
is stated in numbers rather than in prose. The intensity value itself carries nothing: any single
value reaches the same guard.

**Not interchangeable with `bench_cube2_constant`.** That cube is 2x2x2, so every voxel has the same
26-neighbourhood count and PyRadiomics' dependence matrix over it is a single cell. This one spreads
the same eight-voxel idea over four dependence values, which is what lets the negative control below
check the dependence axis rather than one number.

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

---

## `bench_shape8_concave_holed` — one concave 26-pixel ROI with an interior hole

| | |
|---|---|
| Data | `shape2d_morphology_mask` + `shape2d_morphology_intensity` in `tests/test_data.h`, loaded by `load_masked_test_roi_data()` at `make_shape2d_settings()`; built for this family by `build_radial_2d_roi()` in `tests/test_2d_radial_common.h`, and for Zernike by `build_zernike_2d_roi()` in `tests/test_2d_zernike_common.h` |
| ROI | one of them — label 1, **26 pixels**, total intensity **1048**, per-pixel intensity 12–68 |
| Shape | an 8×8 grid; the ROI spans x 0–5, y 0–6 as a concave blob with a single one-pixel interior hole at (3,3) |
| Why it exists | a compact 2D shape that is neither convex nor simply connected, so contour tracing has to return two contours and every shape descriptor computed from them is non-degenerate |

Recipes: `radial.shape2d_native`, `morphology.shape2d_native`, `radial.cellprofiler_8bin`,
`morphology.cellprofiler_edge_intensity`, `zernike.shape2d_native`.

Tests reaching it today: `test_2d_radial_{regression,invariant,mechanics}.h` (through
`test_2d_radial_common.h`), `test_2d_morphology_common.h` and the morphology files that include it,
`test_2d_zernike_{analytic,regression,invariant,mechanics}.h` (through `test_2d_zernike_common.h`).

**The interior hole is the load-bearing property, and it is also what limits this fixture.** It is
what makes `LR::merge_multicontour` concatenate two contours rather than return one, which is the
condition under which `Pixel2::find_center` and `Pixel2::max_sqdist` return non-extremal answers
(`audit/radial_2d_cellprofiler_vetting_report.md` §6 defect 2, pinned in
`test_2d_radial_mechanics.h`). For the same reason it is **not** a fixture the radial family can ever
be vetted on: its distance-to-edge maximum is attained by 8 of its 26 pixels, so CellProfiler's own
centre moves with the label image's padding, and 26 pixels over 8 radial bins leaves 3 of them empty
on one side or the other. `audit/radial_2d_golden_regen.md` §5 lists what a vetting fixture would
have to add.

---

## `bench_imq_quality_roi` — one 8×12 image-quality ROI

| | |
|---|---|
| Data | `im_quality_intensity` + `im_quality_mask` in `tests/test_data.h`, loaded by `calc_imq_feature()` in `tests/test_imq_common.h` |
| ROI | one — the mask is all 1s, so the ROI image matrix is the whole bounding box |
| Shape | 8 wide × 12 tall, 96 pixels, grey values {0, 1, 4, 6} |
| Why it exists | the only fixture the IMQ family runs on; it is small enough to featurise inline and carries the vertical stripe structure the Laplacian responds to |

Recipes: `imq.laplacian_ksize1_zeropad`, `imq.saturation_observed_extremum`,
`imq.regression_quality_roi`.

Tests reaching it today: `test_imq_{opencv,cellprofiler,regression}.h`.

**The 0s are a fixture typo that three features now depend on.** Rows y=7..9 of the intensity
literal repeat the *coordinates* of rows 1..3 for x=3..8, so those 18 positions are never assigned
and stay background 0. That makes 0 the ROI's observed minimum, which is what `MIN_SATURATION`
counts (18/96), and it puts a hard edge in the middle of the image that both focus scores respond
to. Fixing the typo would move four goldens, so it is recorded here rather than repaired.

**Its size is what leaves `POWER_SPECTRUM_SLOPE` pinned at its guard.** `rps()` returns early unless
`floor(min(h,w)/8) >= 3`, i.e. unless the short side is at least 24 px. At 8 wide this fixture never
reaches the radial-binning code, and the pinned 0 is the guard's return value. The larger ROI that
feature needs lives in `bench_imq_matrix_cell_rois` below.

## `bench_imq_matrix_cell_rois` — three probe ROIs for the cells outside the 8×12 fixture

| | |
|---|---|
| Data | built in `tests/test_imq_regression.h` (`imq_constant_roi_intensity`, `imq_narrow_mask_intensity` + `imq_narrow_mask_mask`, `imq_large_roi_intensity`), fed through `calc_imq_feature_on()` |
| ROIs | three, one per matrix cell — a constant 4×4, a 4×4 whose mask covers 5 of the 16 pixels in its bounding box, and a 24×24 modular ramp `1 + (7x + 13y) mod 64` |
| Why it exists | `bench_imq_quality_roi` cannot reach these cells at all: its mask fills its bounding box, its extrema differ, and it is 8 px wide. Each ROI here is the smallest input that reaches one VALID-BUT-PRODUCTION-ONLY cell of `matrix/imq.md` |

Recipes: `imq.saturation_production_only`, `imq.power_spectrum_past_guard`.

Tests reaching it today: `test_imq_regression.h`.

**They are built rather than added to `test_data.h`.** Each is a closed-form pattern a reader can
evaluate by hand — the saturation pins are `k/16` for a countable `k`, and the ramp is a one-line
formula — so a literal pixel table would be less checkable than the loop that generates it, not more.

**The 24×24 is the smallest ROI past the power-spectrum guard, and it is deliberately textured.**
`floor(min(h,w)/8) >= 3` first holds at 24. A smooth ramp there leaves fewer than two points
surviving `magnitude[i] > 0 && isfinite(log(raw_power[i]))`, `power_spectrum_slope()` falls through
to the same `0` the guard returns, and the pin could no longer tell the two paths apart. The modular
ramp leaves 3 surviving points and a fitted slope of 1.7837481542489078.

---

## `bench_disk64_diagonal_boundary` — one 64×64 disk, for the contour production cells

| | |
|---|---|
| Data | built in `tests/python/test_2d_ooc_invariant.py`, `test_2d_ooc_regression.py` and `test_2d_morphology_regression.py`: mask `(y-32)² + (x-32)² <= 400`, intensity `(1 + x + 7y)` inside it, written as a TIFF pair per test |
| ROI | one — **1257 pixels**, intensity 117–397, total 323049; **112** of them are edge pixels under the 4-neighbour inner boundary |
| Shape | a filled disk of radius 20 centred in a 64×64 frame |
| Why it exists | its boundary is genuinely **diagonal**, which is what the other out-of-core fixtures are not |

Recipes: none — this fixture backs `regression` and `invariant` assertions only, never an oracle row.

Tests reaching it today: `test_2d_ooc_invariant.py`
(`test_2d_ooc_2d_contour_intensity_matches_in_ram_on_diagonal_boundary_invariant`),
`test_2d_ooc_regression.py`, `test_2d_morphology_regression.py`.

**The diagonal boundary is the load-bearing property.** Every other `_ooc_` fixture is a full-image
rectangle, and around a rectangle every contour step is an axis-aligned unit step — so the contour
pixel *count* and the sum of Euclidean step lengths are the same number, and the two contour
implementations agree by construction rather than by correctness. On this disk they separate: the
in-RAM path returns 131.88225099390849 and the out-of-core path returns **112.0**, which is exactly
the edge-pixel count in the row above. That is what identifies the divergence as a difference of
definition rather than of accumulation, and it is why `matrix/morphology.md` marks the out-of-core
`PERIMETER` cell a defect.

**It is built rather than added to `test_data.h`** because it is a one-line closed form a reader can
evaluate by hand, and because a 4096-pixel literal table would be far less checkable than the two
lines that generate it.
