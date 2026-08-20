# Not covered — tests the registry does not reference, and tests that never run

Baseline for the family-by-family reorg (SPEC §9 rollout). Two questions, answered mechanically
against the tree at the head of this branch:

- **A.** which test files does `oracle_coverage.csv` not reference at all?
- **B.** which tests never execute — because nothing includes them, nothing registers them, or a
  build flag excludes them?

Both are exact: A is a set difference over the `current_test` / `target_test` columns, B reads
`test_all.cc`'s `#include` closure and its `TEST()` registrations. Neither relies on matching feature
names into files — that was tried and it produces false results in both directions (GT tables key
their goldens inconsistently, e.g. the IBSI intensity-histogram table uses `"VARIANCE_IDX"` with the
`IH_` prefix dropped, while oracle helpers reach features indirectly).

Regenerate by re-deriving both sets; keep this file in step with each reorg wave.

---

## A. Test files no registry row references — 24 files, 58 test functions

### A.1 Correctly absent — plumbing, fixtures and framework self-tests (17 files)

These assert no feature value, so they have no `(feature × config × oracle)` row by construction
(SPEC §1). Recording them here so their absence is a documented decision, not an oversight.

| file | functions | why it has no row |
|---|---:|---|
| `test_3d_nifti_mechanics.h` | 2 | NIfTI loader geometry / data access |
| `test_arrow_mechanics.h` | 2 | Arrow + Parquet writer plumbing |
| `test_arrow_file_name_mechanics.h` | 1 | output-file naming rules |
| `test_2d_glcm_mechanics.h` | 1 | guards the `GLCM_OFFSET` default (a setting, not a value) |
| `test_initialization_mechanics.h` | 1 | environment init |
| `test_2d_omezarr_mechanics.h` | 6 | OME-Zarr tile/raw loader |
| `test_roi_blacklist_mechanics.h` | 1 | ROI blacklisting |
| `test_2d_tiff_loader_mechanics.h` | 1 | uint32 strip loader |
| `test_2d_ooc_invariant.py` | 9 | out-of-core == in-RAM equality; spans all features, per-feature rows would be meaningless |
| `test_2d_ooc_mechanics.py` | 1 | oversized-montage failure path |
| `test_vetting_mechanics.py` | 5 | self-test of `check_coverage.py` |
| `test_3d_coverage_common.h`, `test_3d_morphology_common.h`, `test_2d_moments_common.h`, `test_2d_morphology_common.h` | 0 | shared fixtures; the kind belongs to the files that include them |
| `test_feature_calculation_common.h` | 0 | shared `test_feature` template helper |

### A.2 Assert feature values but no row lists them — real gaps (7 files)

These do produce feature numbers, so a `current_test` entry is missing. Ordered by how much the
omission matters:

| file | functions | what it asserts | why it matters |
|---|---:|---|---|
| ~~`test_neighbors_oracle.py`~~ | 2 | `PERCENT_TOUCHING`, `NUM_NEIGHBORS`, closest-neighbor distance on the production `featurize()` path | **CLOSED in Wave 15.** On inspection it asserts bounds and relations (`<= 100`, `> 0`, `== 100` for an enclosed ROI), not oracle values — so it is an `_invariant`, not the CellProfiler oracle assumed here. Renamed `test_2d_neighbor_invariant.py` and added to `current_test` on the 3 rows it covers. |
| `test_2d_hu_regression.py` | 3 | first-order MIN/MAX/MEAN/INTEGRATED in `--preserve-hu` mode | a second config for existing firstorder rows (SPEC §1 "vetted on config A") — no row records it |
| `test_3d_hu_nifti_regression.py` | 3 | the same on a 3D NIfTI volume with `scl_slope`/`scl_inter` | ditto, 3D |
| `test_hu_analytic.h` | 3 | closed form of the HU offset mapping (`uint_friendly_inten`) | analytic assertion with no row |
| `test_2d_hu_mechanics.h` | 8 | loader-level HU preservation (TIFF / DICOM / float) | plumbing, but it pins values |
| `test_2d_signed_int16_loader_mechanics.py` | 2 | MIN/MAX/MEAN do not wrap for signed int16 | guards a wrap bug that silently corrupted values |
| `test_2d_tiff_loader_mechanics.py` | 2 | pixel values and feature equality for uint32 strip TIFFs | guards a heap over-read that corrupted values |
| `test_2d_contour_analytic.h` | 5 | contour tracing (pixel counts / connectivity) | underlies `PERIMETER`, but asserts geometry rather than a feature |

**Decision needed per row:** either add these files to `current_test` for the features they touch, or
state in `notes` that the HU / loader configs are deliberately out of the per-feature registry.

### A.3 The mirror image — registry references to files that do not exist

`target_test` is the reorg destination, so a dangling entry is **backlog, not error**: it names where
an assertion is to be written or moved. **107 refs across 9 filenames** (was 204 across 14, itself
down from 256 across 17). The table below is recomputed from the registry rather than carried
forward, because the previous one had gone stale in both directions: it listed files that meanwhile
came into existence (`test_3d_glcm_regression.h`, `test_2d_intensity_histogram_analytic.h`,
`test_3d_gldzm_regression.h`, `test_2d_morphology_cellprofiler.h`) and counted `.py` targets as
missing because it looked only in `tests/`, not `tests/python/`. `test_2d_glcm_matlab.h` left the
list a third way — the ten rows that named it were demoted to `regression` and repointed at
`test_2d_glcm_regression.h` (see section C).

| target file named by the registry | rows waiting | family |
|---|---:|---|
| `test_2d_glcm_pyradiomics.h` | 28 | glcm |
| `test_2d_glrlm_pyradiomics.h` | 20 | glrlm |
| `test_2d_ngldm_mirp.h` | 19 | ngldm |
| `test_2d_gldzm_mirp.h` | 17 | gldzm |
| `test_2d_glszm_pyradiomics.h` | 10 | glszm |
| `test_2d_ngtdm_pyradiomics.h` | 5 | ngtdm |
| `test_3d_glcm_mirp.h` | 5 | glcm (3D) |
| `test_3d_glrlm_mirp.h` | 2 | glrlm (3D) |
| `test_3d_ngldm_mirp.h` | 1 | ngldm (3D) |

One genuine error, separate from the backlog: **`test_glcm.h` appeared in `current_test` on 3 rows**
although Wave 2 renamed it to `test_2d_glcm_regression.h`. **CLOSED in the glcm wave — but not by
repointing it.** The three rows are the *first-order* `ENERGY`, `ENTROPY` and `VARIANCE`, and
`test_2d_glcm_regression.h` asserts only `Feature2D::GLCM_ENERGY`/`GLCM_ENTROPY`/`GLCM_VARIANCE`, never
the bare first-order features. The reference was a name-collision artifact of the original audit scan,
so it was removed rather than renamed; each row already lists the first-order files that do assert it.
`current_test` must track renames — but a reference has to be checked for meaning before it is
carried forward, or a rename wave will preserve a claim that was never true.

Per family, how much of the destination map is already in place:

| family | rows | target exists | dangling | no target set |
|---|---:|---:|---:|---:|
| gldm ✅ | 28 | **28** | 0 | 0 |
| moments | 180 | **180** | 0 | 0 |
| neighbor | 9 | **9** | 0 | 0 |
| imq | 6 | **6** | 0 | 0 |
| radial / zernike / gabor | 3 / 1 / 1 | all | 0 | 0 |
| morphology | 113 | 107 | 6 | 0 |
| glrlm | 64 | 42 | 22 | 0 |
| glszm | 32 | 22 | 10 | 0 |
| ngldm | 38 | 18 | 20 | 0 |
| firstorder ✅ | 72 | **72** | 0 | 0 |
| glcm | 118 | 38 | 80 | 0 |
| ngtdm | 10 | 5 | 5 | 0 |
| gldzm | 36 | 1 | 35 | 0 |
| intensity_histogram | 47 | 1 | 26 | 20 |

**Done so far:** `gldm` (Wave 11, pure rename — all targets already existed) and `firstorder`
(Wave 12, which had to *create* `test_2d_firstorder_matlab.h` and move 26 assertions into it, because
column J named a file that did not exist).

---

## B. Tests that never run — 114 functions plus 13 config-gated cases

### B.1 In files `test_all.cc` never includes — 112 functions, zero execution

Seven 3D snapshot files were written but never wired in (MIGRATION §5.10 "systemic orphan finding").
They compile nowhere, so they cannot even fail.

| file | dead functions |
|---|---:|
| `test_3d_firstorder_matlab.h` | 35 |
| `test_3d_glcm_regression.h` | 25 |
| `test_3d_glrlm_regression.h` | 16 |
| `test_3d_glszm_regression.h` | 16 |
| `test_3d_gldm_regression.h` | 14 |
| `test_3d_ngtdm_regression.h` | 5 |
| `test_3d_firstorder_regression.h` | 1 |

The file count is seven rather than six because `3COVERED_IMAGE_INTENSITY_RANGE` — the only 3D
first-order feature with `status=regression` — was split out of `test_3d_firstorder_matlab.h` into its
own file. Both halves remain unwired, so the total is unchanged at 112.

All seven are listed in `current_test` for their families' rows — so the registry currently credits
coverage to assertions that have never executed. Whether to wire them in or delete them is a
behavioural decision (they may fail on first run) and belongs to each family's wave, not to a rename.

### B.1.1 Wired, registered, and asserting nothing — CLOSED for gldm

A fourth way a test can fail to test: it runs, it is registered, and it compares nothing.
`test_2d_gldm_regression.h` held 14 cases and a 14-entry golden table, and
`assert_gldm_feature_regression()` computed the feature, called `save_value()` and returned — no
comparison. All 14 passed as long as `calculate()` did not throw, and the table was referenced
exactly once, by its own declaration. **Closed:** the comparison was added and the table renamed
`gldm_2d_regression_ref_vals` per §6.3.1.

The pinned numbers did not survive contact: all 14 failed on first run. They dated from 01/18/23
and matched neither current output nor the IBSI table next door, and two different features
(`GLDM_SDE`, `GLDM_LGLE`) carried the identical value `0.419444` — a table nothing reads is a table
nothing corrects. They were re-derived from current output at full precision, so the file is now an
honest drift guard on the `cat2500` fixture, which no other GLDM test covers.

Worth a sweep rather than a spot fix: any `assert_*` helper whose body ends without an
`ASSERT_`/`EXPECT_` on a value has this shape.

### B.2 In wired files but never registered — 2 functions

| file | function |
|---|---|
| `test_3d_glcm_pyradiomics.h` | `test_3d_glcm_jvar_pyradiomics` |
| `test_2d_firstorder_ibsi.h` | `test_2d_firstorder_robust_mean_absolute_deviation_ibsi` |

Both are complete assertions with no `TEST()` entry — a missing registration, most likely an
oversight when the surrounding cases were added. Enabling them may fail, so treat as triage.

Not dead, listed to avoid double-counting: `assert_gldzm_matrix_ibsi`,
`assert_ngldm_matrix_ibsi_mode`, `assert_ngldm_matrix_nonibsi_mode` take no arguments and have no `TEST()`
of their own, but each is called by a registered case — they are helpers wearing a `test_` prefix.

### B.3 Registered but excluded by a build flag — 13 cases

Not dead; they run only where the feature is compiled in. A CPU/tiff-only build silently omits them,
so a green local run is not a green matrix.

| flag | cases |
|---|---|
| `USE_ARROW` | `TEST_ARROW_IPC_MECHANICS`, `TEST_ARROW_PARQUET_MECHANICS`, `TEST_ARROW_FILE_NAMING_MECHANICS` |
| `OMEZARR_SUPPORT` | `TEST_2D_OMEZARR_TILELOADER_{GEOMETRY,CONTENT,MULTITILE}_MECHANICS`, `TEST_2D_OMEZARR_RAW_{GEOMETRY,CONTENT,MULTITILE}_MECHANICS` |
| `DICOM_SUPPORT` | `TEST_2D_HU_LOADER_DICOM_{U16,I16}_PRESERVE_MECHANICS`, `TEST_2D_HU_LOADER_DICOM_CT_SMALL_{PRESERVE,BASELINE}_MECHANICS` |

### B.4 Python tests skipped at runtime

- `test_2d_hu_regression.py`, `test_3d_hu_nifti_regression.py`, `test_2d_hu_ct_small_pydicom.py` — module-level
  `skipif` when their committed fixtures are absent.
- `test_2d_morphology_invariant.py`, `test_2d_glcm_pyradiomics.py`, `test_2d_gldm_pyradiomics.py` — skip when the
  canonical ROI cannot be parsed out of `test_data.h`. **A skip here silently removes an oracle
  assertion**; these should fail rather than skip, since the fixture is committed.
- `test_nyxus.py` — one case is skipped on Python 3.12, one is `skip_ci`.

---

## C. Oracle assertions with no recorded provenance

SPEC §6.4 requires tool + version + exact config + generator path at every pinned oracle golden.
Surfaced by Wave 12 and not yet satisfied:

| site | assertions | what is missing |
|---|---:|---|
| `test_2d_firstorder_matlab.h` | 34 | all 34 are MATLAB values (`oracle_3p_matlab_*` named the tool, `oracle_3p_builtin_*` meant MATLAB's built-ins). Missing: MATLAB version, exact config, generator path — the numbers are here, the reproduction recipe is not |
| `test_3d_firstorder_matlab.h` | 35 | `firstorder_3d_matlab_ref_vals` values also come from MATLAB, but the map says nothing about it. The 36th assertion moved to `test_3d_firstorder_regression.h`, which reads the same map — so the gap covers both files |
| `test_2d_morphology_cellprofiler.h` | 6 | the 5 `EDGE_*` + `MASS_DISPLACEMENT` now have their own `morphology_2d_cellprofiler_ref_vals`, split out of the shared snapshot map they used to be asserted against. That fixes the *name*, not the *evidence*: no CellProfiler version, config or generator is recorded, so nothing in the tree distinguishes a CellProfiler number from a Nyxus one. The registry's `vetted` verdict rests on the tracker alone. Closing it means a `gen_morphology_cellprofiler.py` run; if that is not going to happen, the honest alternative is demotion to `regression`, as the ten GLCM `matlab` rows took |
| `test_3d_morphology_matlab.h` | 3 | `morphology_3d_matlab_ref_vals` (MATLAB R2025b regionprops3) records tool, properties, fixture and recipe, but no generator — the numbers cannot be regenerated from the tree. Bands are measured rather than assumed (`3VOXEL_VOLUME` 2.3e-04%, the two hull volumes 3.58%), so the gap here is reproducibility only |

Closing these means writing tool + version + config + generator down at each assertion site, ideally
by regenerating through the harness so the values become reproducible.

Two grades of gap sit in that table, and they are not equally serious. For the MATLAB rows the values
*are* the oracle's — only the reproduction recipe is absent. For `test_2d_morphology_cellprofiler.h`
even that much is unestablished: nothing in the tree shows the numbers ever came from CellProfiler,
so the entry records an unproven claim rather than an unreproducible one. Do not treat the two the
same way when closing them.

**`test_2d_glcm_matlab.h` was a different case, and is now closed by demotion.** Its own header
declared its provenance gap tracked here, but this section never listed it, and the gap was not the
missing kind. The other entries hold MATLAB values with no recipe; that file held **no MATLAB values
at all**. Its ten functions called `assert_glcm_feature_regression` on the regression file's fixture,
against the regression file's snapshot table, at the regression file's 1% tolerance — nothing but the
names separated them from a regression assertion. The table cannot be MATLAB's, because it pins Nyxus
output and was refreshed in 2026-06 to follow a Nyxus bug fix; and three of the five features
(`ASM`, `ENERGY`, `CORRELATION`) sit in the transpose-sensitive group measured to diverge from a
symmetric-matrix tool by 3.7% and more. The rows read `source=tracker` with no recipe and no
tolerance, so the claim was inherited from a spreadsheet, never executed. The ten assertions moved
into `test_2d_glcm_regression.h` as `_regression` (no coverage lost — no other file asserted them)
and the ten rows now read `status=regression`, `candidate_oracle=matlab (graycoprops)`,
`flag=unproven-reference`.

Re-vetting them needs goldens from `graycoprops` itself — the five map onto its four properties
(Contrast, Correlation, Energy, Homogeneity). Note for whoever picks it up: the Octave harness cannot
do it as it stands. Octave's `image` package ships `graycomatrix` but **not** `graycoprops`, so this
one needs real MATLAB, or a checked-in reimplementation of the four published formulas recorded
honestly as `oracle=analytic` rather than as `matlab`.

**An oracle-suffixed test asserted a feature the registry says has no oracle (2D neighbour).**
`test_2d_neighbor_percent_touching_enclosed_analytic` lived in `test_2d_neighbor_analytic.h` and
asserted `PERCENT_TOUCHING`, whose registry row correctly reads `status=regression` with an empty
oracle. Every coverage scanner in this tree -- `check_test_names.py`, `report_feature_tests.py`, the
per-family `scan_*` scripts -- attributes an oracle from the test function's **name suffix**, so a
scan would have credited the feature with `oracle=analytic`. What the function actually asserts are
bounds (`0 <= PT <= 100`, and exactly 100 for a fully enclosed ROI), which SPEC 4.4 classes as
invariants, not oracle comparisons. Moved to `test_2d_neighbor_invariant.h` under `_invariant` names.
**The name is the claim** -- when a bound assertion needs a home, an oracle file is the wrong one.

**Both 2D neighbour generators validated themselves rather than the headers they feed.** Each carried
a hardcoded copy of the goldens (`PINNED` / `CP_VETS`) and compared its fresh run against that copy,
so editing a golden in either header would not have been caught by anything. `gen_neighbor_analytic.py`
also named the wrong file: its comment and banner both said `test_2d_neighbor_regression.h`, which
those goldens had already been moved out of. Both now parse the header they feed, verify every pin,
report any feature they produce that the header fails to pin, and exit non-zero. See
`audit/neighbor_2d_analytic_vetting_report.md`.

**A helper-block boundary bug over-credited coverage in the per-family scanners.** The shared
`helper_features()` logic bounded a python helper's body at the next *helper* rather than the next
top-level `def`, so the last helper in a file swallowed every test function below it, absorbed every
feature name in the file, and passed the lot to each test that called it. In the 2D neighbour family
that credited a `PERCENT_TOUCHING`-only test with `NUM_NEIGHBORS` and `CLOSEST_NEIGHBOR1_DIST`. Fixed
in `audit/scan_neighbor_coverage.py`; **the same logic is copied in the other per-family `scan_*`
scripts and should be corrected there as each family is re-vetted.**

## D. Registry rows that contradict themselves

Found while placing assertions by column J. Each needs a registry decision, not a code change:

| row | `oracle` (E) | `target_test` (J) | what the tree actually holds |
|---|---|---|---|
| 2D `UNIFORMITY` | pyradiomics | `test_2d_firstorder_regression.h` | **RESOLVED.** A pyradiomics assertion already exists in `test_2d_firstorder_pyradiomics.h`, so column J was stale: repointed there, no code moved. The MATLAB-valued assertion in `test_2d_firstorder_matlab.h` is a *second* assertion at the MATLAB config and needs its own row per SPEC §3. |
| 2D `ENTROPY` | pyradiomics | `test_2d_firstorder_regression.h` | **RESOLVED.** Also already asserted in `test_2d_firstorder_pyradiomics.h`; column J repointed, no code moved. The snapshot in the regression file is the drift guard on the default config. |
| 2D `moments` ×40 | skimage | `test_2d_moments_regression.h` | **RESOLVED in Wave 13.** All 40 were already asserted in `test_2d_moments_skimage.h`; only the registry was stale, so `target_test` was repointed there and no code moved. This is the common case of the class: the assertion was migrated in an earlier wave and column J was never updated. |
| 2D `ngldm` x19 | mirp | `test_2d_ngldm_mirp.h` | The tree holds **IBSI** goldens for 17 of these 19 features, cited page-by-page against the IBSI documentation in `test_2d_ngldm_ibsi.h`; the other two (`GLM`, `DCM`) are explicitly *not* IBSI features and are now snapshots in `test_2d_ngldm_regression.h`. So the in-tree 2D NGLDM oracle is `ibsi`, not `mirp`. Either these rows should read `oracle=ibsi` (already satisfied, 17 of them), or MIRP is wanted as a second opinion per SPEC 3 and the rows stay backlog until it is run. |
| 3D firstorder x18 | matlab | `test_3d_firstorder_regression.h` | **RESOLVED under SPEC 6.2.1** (this reverses the earlier "leave it" call, which assumed column J was authoritative). `firstorder_3d_matlab_ref_vals` holds MATLAB values, so the file is now `test_3d_firstorder_matlab.h` with 35 `_matlab` functions and the 18 rows point there. The 36th, `3COVERED_IMAGE_INTENSITY_RANGE`, is the one regression-only feature and was split into `test_3d_firstorder_regression.h`. Two carry-overs: neither file is `#include`d (B.1), and the same map also covers 17 `oracle=pyradiomics` features which per SPEC 3 need a second (matlab) row each. |

The general form: **a `vetted` row whose `target_test` names a `_regression` file** is asserting that
its oracle evidence lives outside the tree. That is legitimate but should be explicit — a `source` or
`notes` entry saying "external harness" — rather than inferable only by opening the file.

## Summary

| | count |
|---|---:|
| test files no registry row references | 25 (7 of them assert feature values → A.2) |
| functions in those files | 58 |
| registry rows whose `target_test` is not yet written (backlog) | 256 refs / 17 files |
| stale `current_test` refs (rename drift, must fix) | 3 |
| functions that never execute (unwired) | 112 |
| functions that never execute (unregistered) | 2 |
| cases gated by a build flag | 13 |

The reorg waves should close A.2 (add the missing `current_test` entries) and decide B.1/B.2 per
family; A.3 shrinks as each family's target files get written.
