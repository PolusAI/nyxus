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

## A. Test files no registry row references — 25 files, 58 test functions

### A.1 Correctly absent — plumbing, fixtures and framework self-tests (18 files)

These assert no feature value, so they have no `(feature × config × oracle)` row by construction
(SPEC §1). Recording them here so their absence is a documented decision, not an oversight.

| file | functions | why it has no row |
|---|---:|---|
| `test_3d_nifti_mechanics.h` | 2 | NIfTI loader geometry / data access |
| `test_arrow_mechanics.h` | 2 | Arrow + Parquet writer plumbing |
| `test_arrow_file_name_mechanics.h` | 1 | output-file naming rules |
| `test_glcm_mechanics.h` | 1 | guards the `GLCM_OFFSET` default (a setting, not a value) |
| `test_initialization_mechanics.h` | 1 | environment init |
| `test_omezarr_mechanics.h` | 6 | OME-Zarr tile/raw loader |
| `test_roi_blacklist_mechanics.h` | 1 | ROI blacklisting |
| `test_tiff_loader_mechanics.h` | 1 | uint32 strip loader |
| `test_ooc_invariant.py` | 9 | out-of-core == in-RAM equality; spans all features, per-feature rows would be meaningless |
| `test_ooc_mechanics.py` | 1 | oversized-montage failure path |
| `test_vetting_coverage.py` | 5 | self-test of `check_coverage.py` |
| `test_3d_coverage_common.h`, `test_3d_morphology_common.h`, `test_moments_common.h`, `test_morphology_common.h`, `test_remaining2d_common.h` | 0 | shared fixtures; the kind belongs to the files that include them |
| `test_feature_calculation.h` | 0 | shared `test_feature` template helper |

### A.2 Assert feature values but no row lists them — real gaps (7 files)

These do produce feature numbers, so a `current_test` entry is missing. Ordered by how much the
omission matters:

| file | functions | what it asserts | why it matters |
|---|---:|---|---|
| ~~`test_neighbors_oracle.py`~~ | 2 | `PERCENT_TOUCHING`, `NUM_NEIGHBORS`, closest-neighbor distance on the production `featurize()` path | **CLOSED in Wave 15.** On inspection it asserts bounds and relations (`<= 100`, `> 0`, `== 100` for an enclosed ROI), not oracle values — so it is an `_invariant`, not the CellProfiler oracle assumed here. Renamed `test_neighbor_invariant.py` and added to `current_test` on the 3 rows it covers. |
| `test_hounsfield.py` | 3 | first-order MIN/MAX/MEAN/INTEGRATED in `--preserve-hu` mode | a second config for existing firstorder rows (SPEC §1 "vetted on config A") — no row records it |
| `test_hounsfield_nifti.py` | 3 | the same on a 3D NIfTI volume with `scl_slope`/`scl_inter` | ditto, 3D |
| `test_hu_analytic.h` | 3 | closed form of the HU offset mapping (`uint_friendly_inten`) | analytic assertion with no row |
| `test_hu_mechanics.h` | 8 | loader-level HU preservation (TIFF / DICOM / float) | plumbing, but it pins values |
| `test_signed_int16_loader.py` | 2 | MIN/MAX/MEAN do not wrap for signed int16 | guards a wrap bug that silently corrupted values |
| `test_tiff_loader.py` | 2 | pixel values and feature equality for uint32 strip TIFFs | guards a heap over-read that corrupted values |
| `test_contour_analytic.h` | 5 | contour tracing (pixel counts / connectivity) | underlies `PERIMETER`, but asserts geometry rather than a feature |

**Decision needed per row:** either add these files to `current_test` for the features they touch, or
state in `notes` that the HU / loader configs are deliberately out of the per-feature registry.

### A.3 The mirror image — registry references to files that do not exist

`target_test` is the reorg destination, so a dangling entry is **backlog, not error**: it names where
an assertion is to be written or moved. **204 refs across 14 filenames** (was 256 across 17; the gldm
and firstorder waves closed `test_3d_gldm_regression.h`, `test_firstorder_matlab.h` and
`test_3d_firstorder_matlab.h`).

| target file named by the registry | rows waiting | family |
|---|---:|---|
| `test_glcm_pyradiomics.h` | 34 | glcm |
| `test_3d_glcm_regression.h` | 31 | glcm (3D) |
| `test_intensity_histogram_analytic.h` | 26 | intensity_histogram |
| `test_glrlm_pyradiomics.h` | 20 | glrlm |
| `test_ngldm_mirp.h` | 19 | ngldm |
| `test_gldzm_mirp.h` | 17 | gldzm |
| `test_3d_gldzm_regression.h` | 18 | gldzm (3D) |
| `test_glcm_matlab.h` | 10 | glcm |
| `test_glszm_pyradiomics.h` | 10 | glszm |
| `test_morphology_cellprofiler.h` | 6 | morphology |
| `test_ngtdm_pyradiomics.h` | 5 | ngtdm |
| `test_3d_glcm_mirp.h` | 5 | glcm (3D) |
| `test_3d_glrlm_mirp.h` | 2 | glrlm (3D) |
| `test_3d_ngldm_mirp.h` | 1 | ngldm (3D) |

One genuine error, separate from the backlog: **`test_glcm.h` still appears in `current_test` on 3
rows** although Wave 2 renamed it to `test_glcm_regression.h`. `current_test` must track renames —
that is the one registry obligation a rename wave owes.

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
(Wave 12, which had to *create* `test_firstorder_matlab.h` and move 26 assertions into it, because
column J named a file that did not exist).

---

## B. Tests that never run — 114 functions plus 13 config-gated cases

### B.1 In files `test_all.cc` never includes — 112 functions, zero execution

Six 3D snapshot files were written but never wired in (MIGRATION §5.10 "systemic orphan finding").
They compile nowhere, so they cannot even fail.

| file | dead functions |
|---|---:|
| `test_3d_firstorder_matlab.h` | 36 |
| `test_3d_glcm.h` | 25 |
| `test_3d_glrlm_regression.h` | 16 |
| `test_3d_glszm_regression.h` | 16 |
| `test_3d_gldm_regression.h` | 14 |
| `test_3d_ngtdm_regression.h` | 5 |

All six are listed in `current_test` for their families' rows — so the registry currently credits
coverage to assertions that have never executed. Whether to wire them in or delete them is a
behavioural decision (they may fail on first run) and belongs to each family's wave, not to a rename.

### B.2 In wired files but never registered — 2 functions

| file | function |
|---|---|
| `test_3d_glcm_pyradiomics.h` | `test_compat_3glcm_JVAR` |
| `test_firstorder_ibsi.h` | `test_firstorder_robust_mean_absolute_deviation_ibsi` |

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
| `USE_ARROW` | `TEST_ARROW`, `TEST_PARQUET`, `TEST_ARROW_FILE_NAME` |
| `OMEZARR_SUPPORT` | `TEST_OMEZARR_TILELOADER_{GEOMETRY,CONTENT,MULTITILE}`, `TEST_RAW_OMEZARR_{GEOMETRY,CONTENT,MULTITILE}` |
| `DICOM_SUPPORT` | `TEST_HU_LOADER_DICOM_{U16,I16}_PRESERVE`, `TEST_HU_LOADER_DICOM_CT_SMALL_{PRESERVE,BASELINE}` |

### B.4 Python tests skipped at runtime

- `test_hounsfield.py`, `test_hounsfield_nifti.py`, `test_hu_ct_small_pydicom.py` — module-level
  `skipif` when their committed fixtures are absent.
- `test_morphology_invariant.py`, `test_glcm_pyradiomics.py`, `test_gldm_pyradiomics.py` — skip when the
  canonical ROI cannot be parsed out of `test_data.h`. **A skip here silently removes an oracle
  assertion**; these should fail rather than skip, since the fixture is committed.
- `test_nyxus.py` — one case is skipped on Python 3.12, one is `skip_ci`.

---

## C. Oracle assertions with no recorded provenance

SPEC §6.4 requires tool + version + exact config + generator path at every pinned oracle golden.
Surfaced by Wave 12 and not yet satisfied:

| site | assertions | what is missing |
|---|---:|---|
| `test_firstorder_matlab.h` | 34 | all 34 are MATLAB values (`oracle_3p_matlab_*` named the tool, `oracle_3p_builtin_*` meant MATLAB's built-ins). Missing: MATLAB version, exact config, generator path — the numbers are here, the reproduction recipe is not |
| `test_3d_firstorder_matlab.h` | 36 | `d3inten_GT` values also come from MATLAB, but the map says nothing about it, and 18 of these features are `oracle=matlab` while the file name says `_regression` — see §D |

| `test_morphology_cellprofiler.h` | 6 | the 5 `EDGE_*` + `MASS_DISPLACEMENT` read their values from `unvetted_nyxus_regression_shape2d_feature_golden_values` in `test_morphology_common.h` — a map whose name still says snapshot. The registry vets these against CellProfiler; the map carries no version, config or generator, and renaming golden tables tree-wide is a separate pass. |

Closing these means writing tool + version + config + generator down at each assertion site, ideally
by regenerating through the Octave harness so the values become reproducible. The values themselves are
MATLAB's — it is their reproduction recipe that is absent.

## D. Registry rows that contradict themselves

Found while placing assertions by column J. Each needs a registry decision, not a code change:

| row | `oracle` (E) | `target_test` (J) | what the tree actually holds |
|---|---|---|---|
| 2D `UNIFORMITY` | pyradiomics | `test_firstorder_regression.h` | **RESOLVED.** A pyradiomics assertion already exists in `test_firstorder_pyradiomics.h`, so column J was stale: repointed there, no code moved. The MATLAB-valued assertion in `test_firstorder_matlab.h` is a *second* assertion at the MATLAB config and needs its own row per SPEC §3. |
| 2D `ENTROPY` | pyradiomics | `test_firstorder_regression.h` | **RESOLVED.** Also already asserted in `test_firstorder_pyradiomics.h`; column J repointed, no code moved. The snapshot in the regression file is the drift guard on the default config. |
| 2D `moments` ×40 | skimage | `test_moments_regression.h` | **RESOLVED in Wave 13.** All 40 were already asserted in `test_moments_skimage.h`; only the registry was stale, so `target_test` was repointed there and no code moved. This is the common case of the class: the assertion was migrated in an earlier wave and column J was never updated. |
| 2D `ngldm` x19 | mirp | `test_ngldm_mirp.h` | The tree holds **IBSI** goldens for 17 of these 19 features, cited page-by-page against the IBSI documentation in `test_ngldm_ibsi.h`; the other two (`GLM`, `DCM`) are explicitly *not* IBSI features and are now snapshots in `test_ngldm_regression.h`. So the in-tree 2D NGLDM oracle is `ibsi`, not `mirp`. Either these rows should read `oracle=ibsi` (already satisfied, 17 of them), or MIRP is wanted as a second opinion per SPEC 3 and the rows stay backlog until it is run. |
| 3D firstorder x18 | matlab | `test_3d_firstorder_regression.h` | **RESOLVED under SPEC 6.2.1** (this reverses the earlier "leave it" call, which assumed column J was authoritative). `d3inten_GT` holds MATLAB values, so the file is now `test_3d_firstorder_matlab.h` with 36 `_matlab` functions and the 18 rows point there. Two carry-overs: the file is still not `#include`d (B.1), and the same map also covers 17 `oracle=pyradiomics` features plus 1 regression-only one, which per SPEC 3 need a second (matlab) row each. |

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
