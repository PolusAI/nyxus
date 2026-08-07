# Test-Suite Reorganization — Migration Map & Gaps Register

**Status:** Planning document (Phase 1 of 3). **No test code is changed by this document.**
It maps the *current* test suite onto the taxonomy in [`SPEC.md`](SPEC.md) and flags every place
we are missing information or a decision. Companion machine-readable file:
[`oracle_coverage.csv`](oracle_coverage.csv) (one row per feature, 758 rows).

**Three-phase plan (agreed):**
1. **This doc + `oracle_coverage.csv`** — the "what → what" map + gaps register. *(you are here)*
2. **Oracle-choice discussion** — resolve the flagged gaps (pick mainstream oracles; triage suspected bugs). Decisions recorded back into this doc and the CSV.
3. **Code reorg** — moves/renames/splits + scaffold, delivered in phased waves (scaffold → GLCM template → family-by-family), each keeping the gtest + pytest suites green.

---

## 1. Rules used to build the registry

Ground truth is **`featureset.h` (758 real features)**, via the audit. The reviewer trackers
(`nyxus_2d_feature_test_coverage_tracker`, `nyxus_unit_test_quality_tracker`) are richer per-feature
but **treated as slightly out of date** — features present in `featureset.h` but absent from the
trackers are flagged `not-in-tracker`.

| Situation | Registry status | Gap flag |
|---|---|---|
| Mainstream oracle, agrees ≤10% | **vetted** | — |
| Mainstream oracle, **>10% off** | **regression** | `promote-after-deepdive` (+`suspected-bug` for the 20 "Sus", +`convention-mismatch` for benign) |
| Only a non-mainstream tool (DIPlib/mahotas/Centrosome) | **regression** | `research-mainstream-oracle` |
| No direct oracle / "Not available" | **regression** | `research-oracle` (carries tracker's candidate) |
| No assertion at all | **untested** | `needs-test` |
| In `featureset.h`, missing from tracker | audit fallback | `not-in-tracker` |

**Mainstream oracle set** = SPEC §4 tokens + **`skimage`** (scikit-image, accepted). **`cp-measure` → `cellprofiler`.**
SPEC status set only (`vetted`/`regression`/`untested`); the tracker's agreement nuance is preserved
in the CSV's `agreement` column but does not create new statuses.

---

## 2. Headline coverage (this reconciliation)

| Status | Count | of 758 |
|---|---|---|
| **vetted** (mainstream oracle, agrees) | **393** | 52% |
| **regression** (snapshot / >10% off / no-oracle) | **344** | 45% |
| **untested** | **21** | 3% |

Gap-flag breakdown: `promote-after-deepdive` 216 · `research-oracle` 118 · `not-in-tracker` 53 ·
`convention-mismatch` 43 · `needs-test` 21 · **`suspected-bug` 20** · `research-mainstream-oracle` 4.

---

## 3. Per-family migration map (current files → target files)

Naming: `test_[3d_]<family>_<kind>.{h,py}`, one kind per file (SPEC §2). `<kind>` is an oracle token
(vetted) or `regression` / `mechanics` / `invariant`. Row-level detail is in `oracle_coverage.csv`.

| Family | n (v/r/u) | Oracles | Current files | → Target files |
|---|---|---|---|---|
| **moments** | 180 (78/102/0) | skimage | `test_2d_geometric_moments.h` | `test_moments_skimage.h`, `test_moments_regression.h` |
| **glcm** ✅ DONE (Wave 2) | 118 (72/46/0) | pyradiomics, matlab, mirp | `test_glcm.h`, `test_glcm_oracle.h`, `test_ibsi_glcm.h`, `test_compat_3d_glcm.h`, `test_glcm_oracle.py` (+orphan `test_3d_glcm.h`) | `test_glcm_regression.h`, `test_glcm_mechanics.h`, `test_glcm_ibsi.h`, `test_3d_glcm_pyradiomics.h`, `test_glcm_pyradiomics.py` |
| **morphology** | 113 (39/74/0) | matlab, cellprofiler, skimage, imea | `test_shape_morphology_2d.h`, `test_2d_remaining_features.h`, `test_3d_shape.h`, `test_feature_oracle.py`, `test_convex_hull_invariants.py`, `test_fractal_dim_oracle.py` | `test_morphology_matlab.h`, `test_morphology_cellprofiler.h`, `test_morphology_skimage.h`, `test_morphology_imea.h`, `test_morphology_regression.h`, `test_morphology_invariant.py`, `test_3d_morphology_*.h` |
| **firstorder** | 72 (51/21/0) | matlab, pyradiomics | `test_pixel_intensity_features.h`, `test_ibsi_intensity.h`, `test_3d_inten.h`, `test_compat_3d_fo_radiomics.h` | `test_firstorder_matlab.h`, `test_firstorder_pyradiomics.h`, `test_firstorder_regression.h`, `test_3d_firstorder_pyradiomics.h`, `test_3d_firstorder_matlab.h` |
| **glrlm** | 64 (38/26/0) | pyradiomics, mirp | `test_glrlm.h`, `test_ibsi_glrlm.h`, `test_3d_glrlm.h`, `test_compat_3d_glrlm.h` | `test_glrlm_pyradiomics.h`, `test_glrlm_regression.h`, `test_3d_glrlm_pyradiomics.h`, `test_3d_glrlm_mirp.h`, `test_3d_glrlm_regression.h` |
| **intensity_histogram** | 52 (26/5/**21**) | analytic | `test_intensity_histogram.h`, `test_intensity_histogram.py`, `test_2d_remaining_features.h` | `test_intensity_histogram_analytic.h`, `test_intensity_histogram_regression.h` (+ **21 untested → needs decision**) |
| **ngldm** | 38 (20/18/0) | mirp | `test_ibsi_ngldm.h`, `test_3d_ngldm.h` | `test_ngldm_mirp.h`, `test_3d_ngldm_mirp.h`, `test_3d_ngldm_regression.h` |
| **gldzm** | 36 (17/19/0) | mirp | `test_ibsi_gldzm.h`, `test_3d_gldzm.h` | `test_gldzm_mirp.h`, `test_gldzm_regression.h`, `test_3d_gldzm_regression.h` |
| **glszm** | 32 (26/6/0) | pyradiomics | `test_glszm.h`, `test_ibsi_glszm.h`, `test_3d_glszm.h`, `test_compat_3d_glszm.h` | `test_glszm_pyradiomics.h`, `test_glszm_regression.h`, `test_3d_glszm_pyradiomics.h` |
| **gldm** | 28 (14/14/0) | pyradiomics | `test_gldm.h`, `test_gldm_oracle.h`, `test_ibsi_gldm.h`, `test_3d_gldm.h`, `test_compat_3d_gldm.h`, `test_gldm_oracle.py` | `test_gldm_pyradiomics.h`, `test_gldm_regression.h`, `test_3d_gldm_pyradiomics.h`, `test_gldm_pyradiomics.py` |
| **ngtdm** | 10 (10/0/0) | pyradiomics | `test_ngtdm.h`, `test_ibsi_ngtdm.h`, `test_3d_ngtdm.h`, `test_compat_3d_ngtdm.h` | `test_ngtdm_pyradiomics.h`, `test_3d_ngtdm_pyradiomics.h` |
| **neighbor** | 9 (2/7/0) | cellprofiler | `test_neighbors_2d.h` | `test_neighbor_cellprofiler.h`, `test_neighbor_regression.h` |
| **imq** | 6 (0/6/0) | — | `test_image_quality.h` | `test_imq_regression.h` |

GLCM is the recommended **worked template** (spans every kind: 2D+3D, pyradiomics+matlab+mirp+ibsi+regression+python).

---

## 4. Cross-cutting files that need special handling (flag)

These aren't single-family and can't be renamed 1:1 — they must be **split by family** during the reorg:

- **`test_3d_feature_coverage.h`** (213 assertions) — spans *every 3D family*. Split its per-family assertions into the `test_3d_<family>_*` files above.
- **`test_2d_geometric_moments.h`** (180) — all `moments`; largely a straight rename to `test_moments_*`.
- **`test_shape_morphology_2d.h`** (mega, ~56 features) — fans out into 7 morphology targets (matlab/cellprofiler/skimage/imea/regression) by feature.
- **`test_nyxus.py`** (88 assertions) — API/plumbing across families → belongs in **`_mechanics`** files, not oracle/regression.
- **`test_2d_remaining_features.h`** — spans morphology + neighbor + histogram; split by family.
- Harness/fixtures (`test_main_nyxus.h`, `test_data.h`, `test_dsb2018_data.h`, `test_tissuenet_data.py`) and pure I/O mechanics (`test_tiff_loader.*`, `test_omezarr.h`, `test_arrow*.h`, `test_3d_nifti.h`, `test_initialization.h`, `test_roi_blacklist.h`, `test_feature_calculation_common.h`) are **out of the family taxonomy** — keep as-is or move under a `_mechanics` convention (decision below).

---

## 5. Gaps register (drives the Phase-2 oracle discussion)

Each flagged bucket, what it means, and the decision needed. Row-level lists are queryable in
`oracle_coverage.csv` (filter on `flag`).

### 5.1 `suspected-bug` — 20 rows — **highest priority**
Mainstream oracle exists but Nyxus is **>10% off and the reviewer marked it "Sus"** (likely a Nyxus
bug, e.g. the class the `PERCENT_TOUCHING` off-by-one fell into). **Decision:** triage each; most need
a deep-dive + fix, then promote to `vetted`. These are correctness signals, not doc gaps.

### 5.2 `promote-after-deepdive` — 216 rows
Mainstream oracle exists, Nyxus >10% off but not flagged suspicious (`needs-audit`/`benign`).
Regression for now. **Decision:** per family, is the gap a definition/config convention (document +
tighten) or a real error (fix)? Then promote.

### 5.3 `research-oracle` — 118 rows
No direct mainstream oracle identified yet; carries the tracker's `Candidate Next Tool`
(`MIRP/IBSI check`, `document convention`, `no direct built-in`). **Decision:** for each family,
confirm a mainstream oracle or accept as documented regression. *This is where the "do some research
to find an appropriate mainstream tool" step happens.*

### 5.4 `research-mainstream-oracle` — 4 rows
Only a **non-mainstream** tool (DIPlib/mahotas/Centrosome) was proposed. **Decision:** find a
mainstream equivalent or accept regression.

### 5.5 `needs-test` — 21 rows (all 2D intensity-histogram)
No assertion at all. **Decision:** analytic oracle vs a tool (these are dispersion/index variants —
likely analytic or pyradiomics-firstorder-adjacent).

### 5.6 `not-in-tracker` — 53 rows
In `featureset.h` but absent from the (out-of-date) tracker. **Decision:** classify each against a
tool/analytic; confirms the 758-vs-705 count gap.

### 5.7 `convention-mismatch` — 43 rows (benign)
A tool exists and differs >10% but for a known coordinate/definition convention. **Decision:**
document the convention in the oracle file; keep as a documented regression unless a matching config
recipe closes the gap.

---

## 5.8 Deep-dive oracle research — findings (Phase 2, DONE)

Researched every feature that lacked an oracle (143). Outcome: **108 gained a mainstream oracle**
the tracker had missed, 21 are genuinely Nyxus-unique, 12 are closed-form only, 1 niche, 1 sentinel.
Exact metric names + caveats are in `oracle_coverage.csv` (`candidate_oracle`, filter `flag`).

| Cluster | n | Recommended mainstream oracle (exact metric) | Key caveat |
|---|---|---|---|
| **Moments** (weighted + non-wt raw/central/norm/Hu) | 62 | **scikit-image** `moments[_weighted][_central/_normalized/_hu]` | skimage transposes indices (row=i,col=j → Nyxus m_{j,i}); weighted moments center on intensity-weighted centroid; Hu returned raw (not log), 2D-only; normalized NaN for order<2 |
| **3D first-order** (standard) | 12 | **PyRadiomics / MIRP** (native 3D); `3COV`→mirp `stat_cov`, `3MEDIAN_ABSOLUTE_DEVIATION`→mirp `stat_medad` | PyRadiomics kurtosis is +3 vs excess; match fixed-bin-count binning |
| **3D GLCM** | 7 | **PyRadiomics / MIRP** — `DIS`→DifferenceAverage, `HOM1`→Id, `HOM2`→Idm, `SUMVARIANCE`→ClusterTendency, `ENERGY`→JointEnergy(=ASM), `VAR`→mirp `cm_var` | **config-sensitive**: Nyxus GT is asymmetric/1-offset/100-level; pyradiomics is symmetric+13-direction → needs a matching config recipe or it diverges badly |
| **3D shape** | 9 | **PyRadiomics 3D shape** — Sphericity, SphericalDisproportion, Major/Minor/LeastAxisLength, Elongation, Flatness, SurfaceVolumeRatio, Compactness1/2 | Compactness1/2 disabled by default (enable); align surface-mesh + voxel spacing |
| **Intensity-histogram** (IH dispersion/index) | ~17 | **MIRP** (IBSI IH family); `IH_ENTROPY`/`IH_UNIFORMITY` also PyRadiomics | PyRadiomics firstorder is on *non-discretised* values (IBSI intensity-based-stats) → only Entropy/Uniformity match; vet `_IDX` against MIRP, `_VAL` analytic |
| **Radial distribution** (`RADIAL_CV`,`FRAC_AT_D`,`MEAN_FRAC`) | 3 | **CellProfiler** `MeasureObjectIntensityDistribution` (`RadialDistribution_*`) | Nyxus copied verbatim from CP → near-exact; match center def + 8 bins/slices |
| **CIRCULARITY** | 1 | **PyRadiomics** 2D `Sphericity` (formula-identical) | any mismatch = real bug or perimeter-convention |
| **DIAMETER_MIN_ENCLOSING_CIRCLE** | 1 | **OpenCV** `minEnclosingCircle` / **imea** | opencv radius×2 |

**No mainstream oracle → stay analytic/regression (21):** `ROUNDNESS` (Nyxus formula), chord angles
(`MAXCHORDS/ALLCHORDS_*_ANG` — imea gives lengths not angles), `POLYGONALITY*`/`HEXAGONALITY*`
(Nyxus/WIPP-unique), `GABOR` (WND-CHARM area-fraction score, no scalar oracle), neighbor angles
(`CLOSEST_NEIGHBOR*_ANG`, `ANG_BW_NEIGHBORS_*` — CellProfiler's `AngleBetweenNeighbors` is a different
quantity), 3D GLDZM/NGLDM intermediate means (`3GLDZM_GLM/ZDM`, `3NGLDM_GLM/DCM` — only their
variances have oracles), `3HYPERSKEWNESS`/`3HYPERFLATNESS` (scipy `moment` only), `3COVERED_IMAGE_INTENSITY_RANGE` (uses image dynamic range).
**Niche only:** `ZERNIKE2D` → `mahotas.features.zernike_moments` (accept as niche or keep analytic).
**Analytic-trivial (12):** `3INTEGRATED_INTENSITY`, `3P01/25/75/99` (numpy; IBSI has only P10/P90).

**Token-set impact:** research adds **OpenCV** (min-enclosing-circle) to the tools in play; `skimage`
and `cellprofiler` already accepted. **Family fix applied:** `ZERNIKE2D`→`zernike`, `GABOR`→`gabor`,
`RADIAL_*`→`radial` (they were mis-bucketed under `intensity_histogram`).

---

## 5.9 Wave 2 (GLCM) — executed, with map corrections

The GLCM migration (first code wave) is done — 5 live files renamed (history-preserving `git mv`),
`test_all.cc` includes updated, verified locally: **full gtest suite 696/696 pass, GLCM 166/166**.
Three corrections to the original map surfaced during execution:

1. **`test_glcm_oracle.h` was mislabeled** — it's not a pyradiomics oracle, it's a *mechanics* guard
   for the GLCM offset=0 default bug → renamed `test_glcm_mechanics.h` (not `_pyradiomics.h`).
2. **The 2D pyradiomics oracle is the Python test** `test_glcm_oracle.py` → `test_glcm_pyradiomics.py`.
   There is no C++ `test_glcm_pyradiomics.h`/`test_glcm_matlab.h`; the C++ GLCM files are
   regression/mechanics/ibsi snapshots, and the 2D pyradiomics/matlab vetting lives in Python/offline.
   (The registry's `target_test` for those 2D rows is aspirational until such oracle assertions exist.)
3. **`test_3d_glcm.h` is orphaned** — not `#include`d in `test_all.cc`; its 26 `test_3glcm_*` functions
   never run. Left untouched and flagged in the registry `notes` (decision A). Live 3D-GLCM coverage
   comes from `test_3d_glcm_pyradiomics.h` (ex-`test_compat_3d_glcm.h`) + parameterized
   `test_3d_feature_coverage.h`. A later triage decides delete-vs-wire-in.

**Lesson for later waves:** don't trust the file's *name* for its kind, and confirm each file is
actually `#include`d/registered before treating it as live — the audit/tracker inferred kind from
names and missed both the mechanics mislabel and the orphan.

---

## 5.10 Wave 3 (texture families) — executed

All remaining texture families migrated in one batch (rename-in-place): **GLRLM, GLSZM, GLDM, NGLDM,
GLDZM, NGTDM** (GLCM was Wave 2). Verified locally: **full gtest suite 696/696 — no tests dropped**
(the hard invariant this wave was run against). Per-family present: GLCM 166, GLRLM 96, GLSZM 64,
GLDM 116*, NGLDM 59, GLDZM 54, NGTDM 20 (*GLDM filter overlaps NGLDM by substring; full-suite count is
the definitive gate).

Patterns confirmed across the family:
- `test_<fam>.h` → `_regression.h`; `test_ibsi_<fam>.h` → `_ibsi.h`; `test_compat_3d_<fam>.h` →
  `test_3d_<fam>_pyradiomics.h`. GLDM mirrored GLCM exactly (`test_gldm_oracle.h` is a bug-guard →
  `test_gldm_mechanics.h`; `test_gldm_oracle.py` → `test_gldm_pyradiomics.py`).
- **Systemic orphan finding:** `test_3d_<fam>.h` is **orphaned** (never `#include`d, tests never run)
  for **glcm, glrlm, glszm, gldm, ngtdm** (5 families, ~130 dead 3D-regression assertions). But
  **ngldm and gldzm have LIVE `test_3d_*` files** (they were `#include`d) → renamed
  `test_3d_ngldm_regression.h` (initially `_ibsi`; corrected in PR #385, see below) /
  `test_3d_gldzm_ibsi.h`. This inconsistency (some native 3D texture tests
  wired, most not) is flagged for a coverage-gap triage; live 3D texture coverage otherwise comes from
  the `test_3d_<fam>_pyradiomics.h` (ex-compat) files + parameterized `test_3d_feature_coverage.h`.
- `test_3d_gldzm_ibsi.h`'s kind label is provisional (IBSI-phantom-based, not fully confirmed). The
  matching NGLDM file was CORRECTED to `test_3d_ngldm_regression.h` (PR #385): its `d3ngldm_GT` table
  has no provenance, runs on the Nyxus coverage phantom (not the IBSI digital phantom), and disagrees
  with MIRP by up to ~10x, so it is a drift guard, not an `_ibsi` oracle. The `_ibsi` suffix is left
  free for a genuine 3D IBSI-phantom NGLDM oracle. The rename preserved all tests regardless.

**Remaining families for later waves:** moments, morphology, firstorder, intensity_histogram, neighbor,
imq, gabor, zernike, radial — plus the cross-cutting `test_3d_feature_coverage.h` split (§4) and the
mechanics/fixture renames (§6.3).

---

## 5.11 Wave 4 (intensity, moments, neighbor, imq) — executed, with the first real split

Migrated: **firstorder, intensity_histogram, moments, neighbor, imq**. Verified locally:
**full gtest suite 696/696 — no tests dropped.**

- **firstorder** (family not in the filenames): `test_pixel_intensity_features.h` →
  `test_firstorder_regression.h`; `test_ibsi_intensity.h` → `test_firstorder_ibsi.h`;
  `test_compat_3d_fo_radiomics.h` → `test_3d_firstorder_pyradiomics.h`. `test_3d_inten.h` is
  **orphaned** (not `#include`d) → left + flagged (decision A).
- **intensity_histogram** → `test_intensity_histogram_regression.h`; **neighbor** →
  `test_neighbor_regression.h`; **imq** → `test_imq_regression.h`. (Conservative `_regression` labels
  where the C++ file pins goldens; oracle vetting for these is external/offline, refinable later.)
- **moments — first genuine split.** `test_2d_geometric_moments.h` cleanly separated (4 functions,
  4 golden vectors) into three files: `test_moments_common.h` (shared fixture/helpers, `#pragma once`),
  `test_moments_skimage.h` (the two `oracle_3p_*` vectors + `*_verifiable_with_3p_builtin_oracle`
  functions), `test_moments_regression.h` (the two `unvetted_*` vectors + `*_unvetted_no_direct_oracle`
  functions). Byte-for-byte preserved (311→315 lines, +4 for the two header preambles); all 4 TESTs
  still register (696 unchanged). The common-header pattern avoids ODR duplication since both headers
  compile into the single `test_all.cc` TU.

**Still remaining:** the **morphology mega-split** (`test_shape_morphology_2d.h`, the largest), the
cross-cutting `test_2d_remaining_features.h` + `test_3d_feature_coverage.h` splits, `test_3d_shape.h`,
`gabor`/`zernike`/`radial`, and the mechanics/fixture renames (§6.3).

---

## 5.12 Wave 5 (morphology mega-split) — executed

The largest split. `test_shape_morphology_2d.h` (11 functions, 3 golden maps) separated into 6 files
(byte-for-byte preserved, 356→364 non-blank lines = +8 header preambles). Verified: **full gtest
suite 696/696 — no tests dropped.**
- `test_morphology_common.h` — `#pragma once`, all 3 maps + 5 shared helpers.
- `test_morphology_regression.h` — 8 pure-unvetted functions (basic morphology, ellipse, contour,
  misc, radius + the 3 "verifiable_with_3p" functions whose external tool is unnamed → regression).
- `test_morphology_skimage.h` — `convex_hull_features` (CONVEX_HULL_AREA/SOLIDITY = skimage; the one
  interleaved `CIRCULARITY` unvetted assert rides along — flagged).
- `test_morphology_matlab.h` — `extrema_features` (MATLAB regionprops extrema).
- `test_morphology_fraclac.h` — `fractal_dimension_blob512_oracle` (the FracLac box-count oracle).

Functions were cleanly kind-pure at the function level (only convex_hull mixed 1 assert), so no
function had to be split — the common-header holds the shared infra to avoid ODR duplication.

## 5.13 Octave viability for the `matlab` oracle (research)

Investigated `github.com/vjaganat90/external-verif-nyxus` (the external cross-check harness) + Octave.
Finding: MATLAB is used as a *built-in oracle* for only 3 families — first-order (`prctile`/`iqr`/
`skewness`/`kurtosis`), a shape subset (`regionprops` 12 props), and a 4-property GLCM
(`graycomatrix`+`graycoprops`); everything else is `NO_BUILTIN` → already PyRadiomics. **GNU Octave +
`image`+`statistics` packages is a near-drop-in replacement** — the only real gap is `graycoprops`
(~15-line reimplementation); `regionprops`, `graycomatrix`, and the stats functions are all present,
comfortably within the harness's 5% tolerance. **Implication:** the registry's `matlab` rows
(GLCM_CONTRAST/CORRELATION, morphology EXTREMA → `test_morphology_matlab.h`) become vettable
**license-free**. Recommend adding an `octave` oracle token (license-free realization of `matlab`) and
using Octave for those rows when real oracle assertions are wired post-restructure. Install:
`apt install octave octave-image octave-statistics` or conda-forge `octave` + `pkg install -forge
image statistics`; invoke headless via `octave-cli --eval "pkg load image statistics; ..."`.

## 5.14 Wave 6 (`test_2d_remaining_features.h` multi-family split) — executed

The cross-cutting 2D leftover file (7 functions, spanning 3 families that all share one compute pass)
split by **registry `target_test`**, not by source proximity. Verified: **full gtest suite 696/696 —
no tests dropped.**
- `test_remaining2d_common.h` — `#pragma once`; the shared fixture (`make_remaining2d_settings`,
  `calculate_remaining2d_shape_feature_values`, `calculate_remaining2d_polygonality_feature_values`),
  the 3 assert-helper families, and all 4 golden maps. `git mv`d from the old file to preserve history.
- `test_morphology_regression.h` — gained 5 functions: erosion-complement, caliper (feret/martin/
  nassenstein), chord stats, chord angles, polygonality/hexagonality (all `→ test_morphology_regression.h`
  in the registry; oracles are definition-review/none, so regression-snapshot).
- `test_intensity_histogram_regression.h` — gained the radial-distribution function (`FRAC_AT_D`,
  `MEAN_FRAC`, `RADIAL_CV`; oracle = cellprofiler `MeasureObjectIntensityDistribution`, pending wiring).
- `test_zernike_regression.h` — **new** 1-function family file for `ZERNIKE2D` (§6.1: mahotas not
  accepted → analytic/regression).

Function names kept verbatim (`test_remaining2d_*`) so every TEST() in `test_all.cc` still registers;
the split is purely a file-taxonomy move. 46 registry rows had `current_test` repointed off the old
filename. The shared header spans 3 families deliberately — the single compute pass produces
morphology, radial, and zernike features from one ROI, so a per-family common would triplicate it.

**Still remaining:** `test_3d_feature_coverage.h` (the 213-assertion cross-cutting 3D split — biggest),
`test_3d_shape.h`, and the mechanics/fixture renames (§6.3). Then regenerate `coverage_report.md`.

## 5.15 Wave 7 (gabor) — executed

Trivial 1-feature family rename (§6.4). The `.h`/`.cc` pair renamed `git mv`:
`test_gabor.{h,cc}` → `test_gabor_regression.{h,cc}`; `test_gabor_truth.h` (oracle-data fixture)
left unrenamed like `test_data.h`. Updated the `#include` inside the `.cc`, the `test_all.cc`
include, and the two CMakeLists `TEST_SRC` lines (the `.cc` is a real compile unit; note pre-existing
stale header entries there, e.g. `test_pixel_intensity_features.h`, are tolerated by CMake). Registry
`current_test` for GABOR repointed `test_gabor.cc` → `test_gabor_regression.cc`. Verified: **696/696 —
no tests dropped.**

## 5.16 Wave 8 (`test_3d_shape.h` → 3D morphology) — executed

`test_3d_shape.h` (10 functions, shared `d3shape_GT` map + `test_3shape_feature` helper) split by
registry `target_test`. `get_3d_segmented_phantom()` is defined once in the live
`test_3d_glcm_pyradiomics.h` and only forward-declared `static` elsewhere in the single `test_all.cc`
TU, so the common header keeps the forward declaration. Verified: **696/696 — no tests dropped.**
- `test_3d_morphology_common.h` (`git mv` from `test_3d_shape.h`): `#pragma once`; `d3shape_GT`, the
  `get_3d_segmented_phantom` forward decl, and the `test_3shape_feature` helper.
- `test_3d_morphology_regression.h` — 8 self-referential-snapshot shape features (area, area2volume,
  compactness1/2, spherical_disproportion, sphericity, volumeconvhull, voxelvolume).
- `test_3d_morphology_matlab.h` — `3MESH_VOLUME` (registry matlab/vetted) + the covariance/eigenvalue
  math test (`Pixel3::calc_cov_matrix` / `Nyxus::calc_eigvals`) whose GT is MATLAB `cov()`/`eig()`.

9 registry rows repointed (they also list `test_3d_feature_coverage.h`, split in Wave 9).

## 5.17 Wave 9 (`test_3d_feature_coverage.h` per-family split) — executed

The biggest and most delicate migration. This file is not a per-family oracle test but a **parameterized
completeness sweep**: `INSTANTIATE_TEST_SUITE_P` over all 213 user-facing 3D features (94 with an embedded
3rd-party oracle + 119 local-regression), plus a global count-guard. Split per family (§6.5):
- `test_3d_coverage_common.h` (`git mv` from `test_3d_feature_coverage.h`): the whole shared harness
  (`build_computed_3d_feature_values` cache, GT maps, assert helpers), the two `TestWithParam` fixtures
  and their `TEST_P` bodies, the `TEST_3D_FEATURE_COVERAGE_COUNTS` guard, plus new
  `feature_3d_family_table()` / `family_of_3d_feature()` / `feature_3d_cases_for_family()` helpers that
  classify each feature by first-match on the calculator featuresets.
- `test_3d_<family>_coverage.h` × 9 (glcm, gldm, gldzm, glrlm, glszm, ngldm, ngtdm, morphology,
  firstorder): each re-instantiates the two suites filtered to its family, with a unique prefix. Empty
  (family, kind) pairs (e.g. `GLDM_UNVETTED`, `GLDZM_EMBEDDED`) compile to zero tests without error.

First-match classification guarantees each feature lands in exactly one family, so the 94+119+1 split is
preserved. Verified: the per-family instance counts (firstorder 36, glcm 59, gldm 14, gldzm 18, glrlm 32,
glszm 16, morphology 14, ngldm 19, ngtdm 5 = 213) match the registry `family` column exactly; **full
gtest suite 696/696 — no tests dropped**; 213 registry rows repointed.

**Build-infra fix (long-latent, unblocked this wave):** `tests/CMakeLists.txt` `TEST_SRC` listed
individual `test_*.h` headers, including `test_pixel_intensity_features.h` which was renamed away in
Wave 4. Headers are pulled via `#include`, never compiled from the source list, so the stale entry sat
dormant until a CMakeLists edit forced a reconfigure — then it hard-failed ("Cannot find source file").
Removed all cosmetic `.h` entries, leaving only real compile units (`test_all.cc`,
`test_gabor_regression.cc`, `${TEST_SOURCE_FILES}`). This makes the remaining renames (§6.3) safe. (Note:
because reconfigure had been silently failing, Waves 7–8 were only genuinely compiled once this fix
landed; the current tree builds clean and passes 696/696.)

**Still remaining:** the mechanics/fixture renames (§6.3). Then regenerate `coverage_report.md`.

## 5.18 Wave 10 (mechanics/fixture renames, §6.3) — executed

Renamed the I/O + plumbing tests to the `_mechanics` suffix (`git mv`, headers only; the test function
names and their `TEST()` registrations are unchanged, so nothing drops):
`test_initialization`, `test_roi_blacklist`, `test_tiff_loader`, `test_3d_nifti`, `test_omezarr`,
`test_arrow`, `test_arrow_file_name` → `*_mechanics.h`. Only the seven `#include` lines in `test_all.cc`
changed. Pure fixtures/harness (`test_data.h`, `test_main_nyxus.h`, `test_dsb2018_data.h`,
`test_tissuenet_data.py`) left unrenamed per §6.3. None of these files appear in the registry
`current_test` (they are plumbing, not feature-oracle rows), so no repoint was needed. Verified: **full
gtest suite 696/696 — no tests dropped.**

Note: `test_arrow_mechanics.h` / `test_arrow_file_name_mechanics.h` are under `#ifdef USE_ARROW`, which is
OFF in the local `nyxus-vet` build, so their rename is correct-by-inspection but not compile-exercised
here. The `test_nyxus.py` API-assertion split (§6.3) is a Python-side follow-up, not done in this wave.

All C++ test files now follow the `test_[3d_]<family>_<kind>.{h}` / `test_<area>_mechanics.h` taxonomy.
`coverage_report.md` regenerated.

## 5.19 Baseline for the function-name waves — `not_covered.md`

Waves 2–10 conformed the *file* names; the *function* names still carry the pre-spec vocabulary
(`test_ibsi_gldm_sde`, `test_compat_3GLDM_DE`, `test_3gldm_sde`), so a reader cannot tell an oracle
assertion from a snapshot without opening the file. The remaining waves fix that family by family,
driven by `oracle_coverage.csv` (§6.2).

Before starting, two exact inventories were taken and recorded in
[`not_covered.md`](not_covered.md): test files no registry row references (25 files / 58 functions, of
which 7 do assert feature values and are genuine `current_test` gaps), and tests that never execute
(112 functions in six never-`#include`d files, 2 defined-but-unregistered, 13 build-flag-gated). That
document also tabulates, per family, how much of the `target_test` destination map already exists —
which is how the family order for these waves is chosen.

## 5.20 Wave 11 (gldm function names, §6.2) — executed

`gldm` first because all 28 of its registry rows already name existing target files
(`test_gldm_ibsi.h` for the 14 2D `oracle=ibsi` rows, `test_3d_gldm_pyradiomics.h` for the 14 3D
`oracle=pyradiomics` rows), so no destination had to be invented and the wave is a pure rename.
Verified: **full gtest suite 722/722 — no tests dropped**, 43 gldm cases run (14 ibsi + 14 regression
+ 14 pyradiomics + 1 mechanics); pytest 82 passed / 7 pre-existing Arrow failures / 1 skipped,
unchanged from the base.

- **43 functions renamed** to `test_[3d_]gldm_<subject>_<kind>`: `test_ibsi_gldm_sde` →
  `test_gldm_sde_ibsi`, `test_gldm_sde` → `test_gldm_sde_regression`, `test_compat_3GLDM_DE` →
  `test_3d_gldm_de_pyradiomics`, `test_3gldm_sde` → `test_3d_gldm_sde_regression`,
  `test_gldm_bug_background_excluded` → `…_mechanics`, and the pytest case →
  `test_gldm_background_not_counted_pyradiomics`. The `compat` prefix is dropped because the kind
  suffix now carries what it meant (the PyRadiomics comparison).
- **gtest case names** are `UPPER(function)` for all 43.
- **5 helpers off the `test_` prefix**, since only a registered zero-arg function is a test:
  `assert_gldm_feature_ibsi`, `assert_gldm_feature_regression`,
  `assert_3d_gldm_feature_pyradiomics`, `assert_3d_gldm_feature_regression`.
- **1 file rename:** `test_3d_gldm.h` → `test_3d_gldm_regression.h` (§6.1 — it carried no kind).
  Registry `current_test` repointed on all 28 rows.
- **Left for triage, recorded in `not_covered.md` §B.1:** that file is still not `#include`d in
  `test_all.cc`, so its 14 snapshot assertions have never run. Wiring them in is a behavioural change
  (they may fail), so it is not part of a rename wave.

## 5.21 Wave 12 (firstorder — placed by registry column, not by inertia) — executed

The rule this wave applied, stated explicitly because Wave 11 did not apply it fully: for every row,
**`status` (D) + `oracle` (E) decide the function suffix, and `target_test` (J) decides the file**.
Where an assertion already existed in the wrong file, it was *moved*, not just renamed. Verified:
**gtest 722/722 — no tests dropped**, 58 firstorder cases (unchanged), pytest unchanged.

Placement before → after, for the 72 firstorder rows:

| file | before | after | why |
|---|---:|---:|---|
| `test_firstorder_matlab.h` | did not exist | **26** | the 33 2D `oracle=matlab` rows name it in column J |
| `test_firstorder_pyradiomics.h` | 1 | **2** | `ROBUST_MEAN_ABSOLUTE_DEVIATION` names it |
| `test_firstorder_ibsi.h` | 13 | 13 | already correct |
| `test_firstorder_regression.h` | 28 | **1** | only `ENTROPY` names it in column J |
| `test_firstorder_common.h` | — | fixture | shared `calculate_pixel_intensity_feature_values`, so an oracle file never includes a regression file |

- **78 functions renamed** to `test_[3d_]firstorder_<subject>_<kind>`: `test_ibsi_mean_intensity` →
  `test_firstorder_mean_ibsi`, `test_pixel_intensity_mean` → `test_firstorder_mean_matlab`,
  `test_3inten_cov` → `test_3d_firstorder_cov_matlab`, and the 17 inline 3D cases →
  `TEST_3D_FIRSTORDER_<F>_PYRADIOMICS`. 19 case names whose word order differed from their function
  (`TEST_IBSI_INTENSITY_MEAN`, `TEST_PIXEL_INTENSITY_MAD`, …) were corrected to `UPPER(function)`.
- **The six oracle goldens that were hiding in the regression file** are the reason this wave moved
  code rather than only renaming: `oracle_3p_matlab_uniformity_feature_golden_value` (asserted at the
  1% cross-tool tier with the comment "vs MATLAB") plus five `oracle_3p_builtin_*` constants for
  HYPERSKEWNESS, HYPERFLATNESS, UNIFORMITY_PIU, COVERED_IMAGE_INTENSITY_RANGE and ROBUST_MEAN — all
  five of which are in the registry's 33 `oracle=matlab` rows. Third-party reference values in a
  file whose name claims "snapshot" is exactly the mislabelling SPEC §1 forbids.
- `test_3d_inten.h` → `test_3d_firstorder_matlab.h` (§6.1, and column J of 19 3D rows).
- `current_test` repointed on 36 rows + the 125 `test_3d_inten.h` references.
- **firstorder is now fully placed: 72/72 rows have an existing `target_test`** (was 20/72). The
  tree-wide backlog drops from 256 dangling refs to 204.

**`test_firstorder_matlab.h` follows the `test_firstorder_ibsi.h` trait.** The goldens arrived as six
ad-hoc `static constexpr double oracle_3p_{matlab,builtin}_<feature>_feature_golden_value` constants
plus 28 literals inlined in test bodies, each body rebuilding the same canonical ROI. Restructured to
match the IBSI file exactly: **one keyed map** (`matlab_reference_firstorder_feature_golden_values`,
34 entries), **one `assert_firstorder_feature_matlab()` helper** that computes the ROI once, and **one
one-line test per feature**. Consequence: the 26 multi-feature functions became **34 per-feature
cases** (`min_max_range` → `min` / `max` / `range`, `percentiles_iqr` → `p01`…`p99` +
`interquartile_range`), so the suite goes **722 → 730**. Nothing was dropped; assertion count is
unchanged and each feature is now individually named and reportable, matching the registry's
one-row-per-feature model.

Two features keep an explicit config at the assertion site, which is what SPEC §5 calls a config
recipe — collapsing them onto the default fixture made them fail, and that failure is the reason the
recipe is now recorded in the test rather than implied:
- `UNIFORMITY` is histogram-based, so MATLAB only matches at `GREYDEPTH=20` with the IBSI path off.
- `COVERED_IMAGE_INTENSITY_RANGE` is a fraction of the **slide** dynamic range, so the fixture needs
  slide props (slide 0 spanning 0..65535).

**These are MATLAB values; what is missing is the record, not the origin.** All 34 goldens in
`test_firstorder_matlab.h` came from MATLAB — that is what `oracle=matlab` states, and the pre-rename
constant names said it too (`oracle_3p_matlab_*` named the tool; `oracle_3p_builtin_*` meant MATLAB's
built-in statistics functions). What no golden carries is the SPEC §6.4 record: MATLAB version, exact
config, generator path. The file header says so, and `not_covered.md` §C tracks writing it down —
ideally by regenerating through the Octave harness (§5.13) so the numbers are reproducible rather than
merely trusted. The same holds for `d3inten_GT` in the 3D file.

**One registry row is self-contradictory and was NOT propagated.** `2D UNIFORMITY` reads
`oracle=pyradiomics` with `target_test=test_firstorder_regression.h`, yet the only in-tree golden for
it is MATLAB's (0.0647664, asserted at the 1% tier with the comment "vs MATLAB"). Following column J
literally would have put a MATLAB reference value in a file whose name claims "snapshot" — restating
the mislabelling this wave exists to remove. The assertion therefore stays in
`test_firstorder_matlab.h` as `test_firstorder_uniformity_matlab`, `current_test` records that, and the
row is flagged in `not_covered.md` for reconciliation: either its `oracle` should read `matlab`, or its
`target_test` should name the pyradiomics file and a pyradiomics golden has to be produced.

**Registry-file hazard worth recording:** `oracle_coverage.csv` contains 16 `CR CR LF` sequences and
19 lone `CR`s (present on `main`, not introduced here). Python's `csv` module with `newline=""` reads
those as extra empty records — a full-file rewrite through `csv.writer` therefore *destroys* rows.
Edit this file with line-level substitutions, or read it the way `check_coverage.py` does (plain
`open()`, which yields the correct 758 rows).

## 5.22 Wave 13 (moments) — executed

First of the remaining family-per-commit waves. moments needed no placement work at all: all 180 rows
already name an existing `target_test`, and neither file hides an oracle golden (the regression file's
tables are `unvetted_nyxus_regression_*`, the oracle file's are `oracle_3p_*` from
`gen_moments_skimage.py`). So this is the pure-rename case: 4 functions, no moves.

- `test_2d_shape_geometric_moments_verifiable_with_3p_builtin_oracle` → `test_moments_shape_skimage`
- `test_2d_intensity_geometric_moments_verifiable_with_3p_builtin_oracle` → `test_moments_intensity_skimage`
- `test_2d_shape_geometric_moments_unvetted_no_direct_oracle` → `test_moments_shape_regression`
- `test_2d_intensity_geometric_moments_unvetted_no_direct_oracle` → `test_moments_intensity_regression`
- `test_moments_hu_wedge_skimage` already conformed; all 5 gtest case names are `UPPER(function)`.

Four functions cover 180 features here, the widest assertion-to-feature ratio in the tree — which is why
the family's coverage cannot be read off the test names and has to come from the registry.

**Registry repointed, no code moved.** 40 of the 180 rows are `vetted` by skimage while column J still
named `test_moments_regression.h`. Checking the tree settled it: **all 40 are already asserted in
`test_moments_skimage.h`** — the assertions were migrated in Wave 4 and column J was never updated. So
the fix is a registry repoint of those 40 rows, not a code move, and moments now has zero
contradictions: 118 rows `vetted`→skimage file, 62 `regression`→regression file.

This is what fixed the placement rule for every later wave (SPEC §6.2.1): **columns D+E decide the file**, and
column J is only trustworthy when it agrees with them, because status/oracle were corrected over time
while `target_test` kept its original value. Verified: **gtest 730/730**, 5 moments cases, pytest
unchanged.

## 5.23 Placement rule correction — status+oracle over target_test

Waves 11-12 placed assertions by `target_test` (column J). That is wrong wherever J disagrees with
`status`+`oracle`: the verdict columns were corrected as features were vetted, while `target_test` kept
the value it was seeded with in Phase 1. **A `vetted` row with `oracle=X` belongs in
`test_[3d_]<family>_X`, and a stale J saying `_regression` does not override that** (SPEC §6.2.1).

Measured across the registry: **133 vetted rows sat in a file whose kind contradicts their oracle.**
Applying the rule to the families already migrated brought that to 73, all of which belong to families
whose waves have not run yet (glcm 46, morphology 26, intensity_histogram 1) and will be placed
correctly when they do.

What the rule actually required, once the tree was checked, was mostly *registry* fixes — because the
assertions had already been migrated in earlier waves and only column J was stale:

| rows | expectation | reality | action |
|---|---|---|---|
| moments ×40 (skimage) | move out of the regression file | all 40 already asserted in `test_moments_skimage.h` | repoint J, **no code moved** |
| firstorder 2D ×2 (pyradiomics: ENTROPY, UNIFORMITY) | move to the pyradiomics file | both already asserted there | repoint J, **no code moved** |
| firstorder 3D ×18 (matlab) | move to a matlab file | only assertions are the MATLAB-valued `d3inten_GT` | **renamed** `test_3d_firstorder_regression.h` → `test_3d_firstorder_matlab.h`, 36 functions + helper → `_matlab` |

Two regression rows pointing at *oracle* files were corrected in the same pass (`PERCENT_TOUCHING` →
`test_neighbor_regression.h`, `3COVERED_IMAGE_INTENSITY_RANGE` → `test_3d_firstorder_regression.h`), so
the contradiction count in that direction is now zero.

**The lesson for the remaining waves:** before moving anything, check whether the destination already
asserts the feature — the usual answer is yes, and then the fix is one registry cell, not a code move.
And where the reference tool has no golden in the tree at all (MIRP: 44 rows), nothing moves: a rename
wave cannot manufacture an oracle. Verified after the correction: **gtest 730/730**, registry check
clean, 758 rows intact.

## 5.24 Wave 14 (zernike) — executed

One feature, one function, and a registry that was already self-consistent (`ZERNIKE2D` is
`status=regression` with `target_test=test_zernike_regression.h`, and that is where it lives). The
work was entirely in the names, which claimed the opposite of the registry's decision:

- `test_remaining2d_verifiable_with_3p_builtin_oracle_zernike2d_feature` → `test_zernike_moments_regression`
- `assert_verifiable_with_3p_builtin_oracle_remaining2d_vector_feature` → `assert_zernike_vector_feature_regression`
- `oracle_3p_remaining2d_vector_feature_golden_values` → `nyxus_regression_zernike_vector_golden_values`
- its `SCOPED_TRACE` label `VERIFIABLE_WITH_3P_BUILTIN_ORACLE__` → `REGRESSION__`

The helper and table were safe to rename outright because **`ZERNIKE2D` is their only key and only
caller** — checked before touching them; the near-identical `assert_unvetted_no_direct_oracle_*` twin
serves the radial features and was left alone. This one mattered beyond tidiness: §6.1 rejects mahotas,
the only tool that computes Zernike moments, so ZERNIKE2D has **no** accepted oracle — yet four
identifiers and every failure message said it was third-party-verified. The 0/1 vetted in
`coverage_report.md` and the test names now agree. Verified: **gtest 730/730**.

## 5.25 Wave 15 (neighbor) — executed

All 9 rows already named a consistent `target_test`, so this is a rename plus one placement fix.
Verified: **gtest 730/730**, 5 neighbor cases, pytest unchanged.

- **7 functions renamed** `test_neighborhood2d_*` → `test_neighbor_*_<kind>`, matching the registry's
  family token; the shared fixture builder followed (`calculate_neighbor_feature_values`).
- **One assertion moved:** `test_neighbor_percent_touching_enclosed_analytic` is a closed-form case
  (an ROI enclosed on all sides must be exactly 100% touching) that sat in the regression file →
  `test_neighbor_analytic.h`, joining the other analytic neighbor assertions.
- **`test_neighbors_oracle.py` was misfiled by its own name and by `not_covered.md` §A.2**, which
  called it an oracle-grade test invisible to the registry. It asserts bounds and relations
  (`PERCENT_TOUCHING <= 100`, `> 0` distances, `== 100` for an enclosed ROI) — an **invariant**, not a
  comparison against CellProfiler. Renamed `test_neighbor_invariant.py`, its two functions given the
  `_invariant` suffix, and added to `current_test` on the 3 rows it bounds. The genuine CellProfiler
  oracle is the C++ `test_neighbor_cellprofiler.h`, which was already correct.

Worth noting for later waves: a file named `*_oracle.py` is not evidence of an oracle. Three more
carry that name (`test_feature_oracle.py`, `test_fractal_dim_oracle.py`, `test_glcm/gldm_pyradiomics.py`
were already renamed) and each needs the same read-the-assertions check before its family's wave.

## 5.26 Wave 16 (imq) — executed

All 6 rows already named a consistent `target_test` and each file already held the right kind, so this
was names only. Verified: **gtest 730/730**, 6 imq cases, and the suite count drops **15 → 14**.

- **6 functions gained the family prefix** the rest of the tree uses: `test_focus_score_opencv` →
  `test_imq_focus_score_opencv`, `test_min/max_saturation_cellprofiler` → `test_imq_*`. The two
  snapshots also lost the meaningless `_feature` tail for a kind: `test_power_spectrum_feature` →
  `test_imq_power_spectrum_slope_regression`, `test_sharpness_feature` →
  `test_imq_sharpness_regression`.
- **The ad-hoc `TEST_IMAGE_QUALITY` gtest suite is folded into `TEST_NYXUS`** (§6.2 fixes one suite
  name for the whole tree). It was the only second suite, so `--gtest_filter=TEST_NYXUS.*` had been
  silently skipping all 6 image-quality cases; anyone filtering that way now gets them.
- Four registry `notes` cite gtest case names in prose (e.g. "in test_imq_opencv.h
  (TEST_FOCUS_SCORE_OPENCV)"); those citations were updated with the renames. No structural column
  changed, so `coverage_report.md` is untouched.

This family is also the tidiest illustration of what the suffix buys: 4 of its 6 features are vetted
(2 opencv, 2 cellprofiler) and 2 are snapshots, and the test names now say which is which without
opening a file.

## 5.27 Wave 17 (ngtdm) — executed

First family of the "orphan 3D snapshot + dangling 2D oracle" shape that glszm, glrlm, gldzm and
ngldm all repeat, so it sets the pattern. Verified: **gtest 730/730**, 16 live ngtdm cases
(5 ibsi + 5 regression + 6 pyradiomics).

- **21 functions renamed**: `test_ibsi_ngtdm_<f>` → `test_ngtdm_<f>_ibsi`, `test_ngtdm_<f>` →
  `test_ngtdm_<f>_regression`, `test_compat_3NGTDM_<F>` → `test_3d_ngtdm_<f>_pyradiomics`,
  `test_3ngtdm_<f>` → `test_3d_ngtdm_<f>_regression`, and `test_ngtd_matrix_correctness` →
  `test_3d_ngtdm_matrix_correctness_pyradiomics` (its case name had also lost the trailing `M`).
- **4 helpers → `assert_*`**, one per file.
- **`test_3d_ngtdm.h` → `test_3d_ngtdm_regression.h`** (§6.1: it carried no kind). Still not
  `#include`d, so its 5 assertions remain dead — recorded in `not_covered.md` §B.1, not fixed here.

**The 5 2D rows stay backlog by decision.** All five say `oracle=pyradiomics` with
`target_test=test_ngtdm_pyradiomics.h`, and no such file or golden exists anywhere in the tree — the
2D NGTDM oracle assertions have not been written. Per SPEC §6.2.1 a rename wave cannot manufacture an
oracle, so the assertions stay in `test_ngtdm_ibsi.h` / `_regression.h` where their values actually
come from, and the rows keep pointing at the file that has to be written. Unlike firstorder — where
the MATLAB values were already in the tree and only needed relocating — there is nothing to move.

## 5.28 Wave 18 (glszm) — executed

Same shape as ngtdm at three times the size: 65 functions across four files. Verified:
**gtest 730/730**, 49 live glszm cases (16 ibsi + 16 regression + 17 pyradiomics).

- **65 functions renamed** to `test_[3d_]glszm_<subject>_<kind>` (`test_ibsi_glszm_<f>` → `_ibsi`,
  `test_glszm_<f>` → `_regression`, `test_compat_3glszm_<f>` → `test_3d_glszm_<f>_pyradiomics`,
  `test_3glszm_<f>` → `test_3d_glszm_<f>_regression`), 4 helpers → `assert_*`, and
  `test_glsz_matrix_correctness` → `test_3d_glszm_matrix_correctness_pyradiomics` (its case name had
  also dropped the trailing `M`, as ngtdm's had).
- **`test_3d_glszm.h` → `test_3d_glszm_regression.h`**; still not `#include`d, so its 16 assertions
  stay dead (`not_covered.md` §B.1).

**Two case-name defects found by making case = `UPPER(function)`**, both invisible while the names
were hand-written:
1. **Six cases were partly lowercase** — `TEST_GLSZM_gln`, `TEST_GLSZM_glnn`, `TEST_GLSZM_szn` and
   their `TEST_IBSI_*` / `TEST_COMPAT_3GLSZM_*` twins. A `--gtest_filter` written in the obvious
   uppercase never matched them.
2. **Four cases named a feature that does not exist**: `SALGLZE` / `SAHGLZE`, where `featureset.h` and
   the assertions themselves say `GLSZM_SALGLE` / `GLSZM_SAHGLE`. Failure output would have named the
   wrong feature.

**The 10 2D `oracle=pyradiomics` rows stay backlog** (same as ngtdm): `test_glszm_pyradiomics.h` has
no golden anywhere in the tree, so there is nothing to move.

## 5.29 Wave 19 (glrlm) — executed

Largest family of this shape: 88 functions across four files, with two extras the previous three did
not have — angle-averaged `_ave` variants in both the ibsi and regression files, and a second ibsi
helper for them. Verified: **gtest 730/730**, 72 live glrlm cases (22 ibsi + 32 regression +
18 pyradiomics).

- **88 functions renamed** to `test_[3d_]glrlm_<subject>[_ave]_<kind>`; the `_ave` marker stays in the
  subject position, so `test_ibsi_glrlm_lglre_ave` → `test_glrlm_lglre_ave_ibsi`.
- **5 helpers → `assert_*`**, including the second ibsi one (`assert_glrlm_ave_feature_ibsi`).
- `test_compat_3glrlm_ave_features` → `test_3d_glrlm_ave_pyradiomics` and
  `test_glrl_matrix_correctness` → `test_3d_glrlm_matrix_correctness_pyradiomics`.
- **`test_3d_glrlm.h` → `test_3d_glrlm_regression.h`**; still unwired, 16 assertions stay dead (§B.1).

**The dropped-`M` typo is now confirmed systemic.** All three matrix-correctness cases carried it —
`TEST_3NGTD_MATRIX_CORRECTNESS`, `TEST_COMPAT_3GLSZ_MATRIX_CORRECTNESS`,
`TEST_COMPAT_3GLRL_MATRIX_CORRECTNESS` — each naming a matrix that is not the family's
(`NGTD`/`GLSZ`/`GLRL` instead of `NGTDM`/`GLSZM`/`GLRLM`). Hand-written case names drift; deriving
them as `UPPER(function)` is what surfaced all three.

**22 rows stay backlog**: 20 2D `oracle=pyradiomics` (`test_glrlm_pyradiomics.h`, no golden in the
tree) and 2 3D `oracle=mirp` (`test_3d_glrlm_mirp.h`; MIRP has no goldens anywhere — 44 rows across
four families wait on it).

## 5.30 Wave 20 (gldzm) — executed

37 functions across two files, and the one wave in this series that **renames a file away from an
oracle token**. Verified: **gtest 730/730**, 36 gldzm cases (19 ibsi + 17 regression).

- **`test_3d_gldzm_ibsi.h` → `test_3d_gldzm_regression.h`.** The file already carried a
  `PROVENANCE: UNKNOWN` header from an earlier investigation stating that its values are *not* IBSI
  consensus: they have no recorded tool/version/config, they are computed on the Nyxus coverage
  phantom rather than the IBSI digital phantom (so IBSI values cannot apply), and an independent MIRP
  run disagrees with every one of them — LDE 314.0 vs 11.235, ZDV 79.7 vs 3.246. The registry agrees:
  all 18 3D rows are `status=regression` targeting `test_3d_gldzm_regression.h`. This wave simply
  acted on a note that had been sitting in the file, and the header now explains the current name
  instead of excusing the old one.
- **37 functions renamed**, `test_ibsi_GLDZM_<F>` → `test_gldzm_<f>_ibsi`, `test_3GLDZM_<F>` →
  `test_3d_gldzm_<f>_regression`; 4 helpers → `assert_*`, one of them (`assert_gldzm_matrix_ibsi`)
  a zero-arg helper that was never registered and so had looked like a dead test.

**A third case-name defect class:** 17 of the 2D cases carried a stray `MATRIX_` infix
(`TEST_GLDZM_MATRIX_LDE`, `TEST_GLDZM_MATRIX_ZP`, …) although they assert *feature values*; the
matrix itself has its own `matrix_correctness` case. Two of the family's cases were also inconsistent
with the rest (`TEST_GLDZM_SDE` without the infix). Deriving case = `UPPER(function)` normalised all
18.

**The 17 2D `oracle=mirp` rows stay backlog** — MIRP has no goldens in the tree. Note the asymmetry
this leaves: 2D GLDZM has exactly **one** vetted-and-asserted feature (`GLDZM_ZDV`, ibsi), the other
17 wait on MIRP, and all 18 3D features are snapshots. `coverage_report.md`'s 18/36 for this family is
the registry's own count and is unchanged by the rename.

## 5.31 Wave 21 (ngldm) — executed

42 functions, and the first **split of an oracle file** in this series. Verified: **gtest 730/730**,
40 ngldm cases (19 ibsi + 21 regression).

- **`test_ngldm_regression.h` created.** `NGLDM_GLM` and `NGLDM_DCM` are Nyxus mean-style rows with
  no counterpart in the IBSI NGLDM table — the source says so in a comment (`--not in IBSI--` beside
  the two commented-out entries) and their goldens are pinned Nyxus output in a table already named
  `unvetted_nyxus_regression_*`. They nevertheless sat in `test_ngldm_ibsi.h`. SPEC §2 keeps one kind
  per file, so they moved out with the snapshot table and helper they use; the two functions lose the
  `unvetted_no_direct_oracle` claim phrase for a plain `_regression` suffix.
- **42 functions renamed**; 5 helpers → `assert_*`, two of which
  (`assert_ngldm_matrix_{ibsi,nonibsi}_mode`) were zero-arg helpers never registered as cases, so
  `not_covered.md` §B.2 had counted them as dead tests.
- All 40 case names were already `UPPER(function)` after the function renames — the first family in
  this series with no case-name defect at all.

**Registry discrepancy recorded, not acted on.** All 19 2D rows read `oracle=mirp` targeting
`test_ngldm_mirp.h`, yet the tree holds **IBSI** goldens for 17 of them (cited page-by-page against
the IBSI documentation) and the remaining two are the non-IBSI snapshots above. So the 2D NGLDM
oracle in the tree is `ibsi`, not `mirp`. Per the placement rule the assertions stay where their
values come from; whether those rows should read `oracle=ibsi` (already satisfied) or keep waiting
for a MIRP cross-check is a registry decision — added to `not_covered.md` §D.

## 5.32 Wave 22 (intensity_histogram) — executed

19 functions, and the second oracle-file creation. Verified: **gtest 730/730**, 12 IH/radial cases,
pytest 7/7 in the renamed module.

- **`test_intensity_histogram_analytic.h` created** — column J names it for 26 of the family's rows
  and it did not exist. Two closed-form assertions moved into it out of the `_ibsi` file:
  `test_intensity_histogram_dispersion_robust_analytic` and `..._bin_counts_analytic`. Both carry
  their own hand-derivable fixtures (the robust-dispersion pixel array moved with them), so neither
  depends on the IBSI phantom runner they were sitting beside. The file that keeps the `_ibsi` name
  now holds exactly one assertion, against IBSI consensus values on the IBSI digital phantom.
- **19 functions renamed**, `test_ih_*` → `test_intensity_histogram_*_<kind>`, and both fixture
  runners to `run_intensity_histogram[_ibsi]_fixture`.
- **Two functions are `_mechanics`, not `_regression`**: `..._gate_off_returns_nan` and
  `..._required_predicate` assert gating behaviour (IH features return NaN unless `ibsi=true`), not
  feature values.
- **The radial function moves out of the family's names**:
  `test_remaining2d_unvetted_no_direct_oracle_radial_distribution_features` →
  `test_radial_distribution_regression`. Its three features (`FRAC_AT_D`, `MEAN_FRAC`, `RADIAL_CV`)
  are `family=radial` in the registry, but column J names the IH regression file, so the assertion
  stays where it is and only the name stops claiming otherwise.
- `tests/python/test_intensity_histogram.py` → `test_intensity_histogram_analytic.py`; its 7 methods
  split `_mechanics` (gating, `--mergerois` API) from `_analytic` (the hand-computed histogram) and
  `_invariant` (index features within bin range).

**24 of the 26 analytic rows remain backlog.** The two assertions written here cover a subset; the
rest of the `_IDX`/`_VAL` dispersion variants still need closed-form assertions, which is vetting
work rather than renaming. This family remains the largest gap in the tree: 47 rows, 4 oracle
assertions.

## 5.33 Wave 23 (morphology) — executed

Widest oracle spread in the tree — matlab, skimage, imea, fraclac, analytic and cellprofiler all
appear in one family's 113 rows. 43 functions renamed. Verified: **gtest 730/730**, 37
morphology/contour cases (5 imea + 4 matlab + 2 skimage + 1 fraclac + 6 analytic + 19 regression),
pytest 8/8 across the three renamed modules.

- **3 imea oracle functions moved** out of `test_morphology_regression.h` into
  `test_morphology_imea.h`: the Martin/Nassenstein, Feret and min-enclosing-circle caliper tests,
  13 assertions comparing against a real `imea_ellipse_caliper_oracle` table (imea
  `measure_2d.statistical_length`, dalpha=10, documented tolerance 0.10 with per-diameter residuals
  written out). Third-party reference values in a file named `_regression`, exactly the firstorder
  pattern.
- **`test_contour.h` → `test_contour_analytic.h`** — its goldens are hand-derived contour geometry.
  Two case names also stopped being opaque: `TEST_CONTOUR_MULTI_1`/`_2` →
  `..._MULTI_DISCONNECTED_ANALYTIC` / `..._MULTI_CONNECTED_ANALYTIC`.

**The three `*_oracle.py` files split three different ways** — the read-the-assertions check from
the neighbor wave, applied and again worth it:

| file | asserts | renamed to |
|---|---|---|
| `test_feature_oracle.py` | `MAXCHORDS_MAX_ANG != MIN_ANG` — a relation, no reference value | `test_morphology_invariant.py` |
| `test_convex_hull_invariants.py` | `SOLIDITY <= 1` — a bound | `test_morphology_hull_invariant.py` |
| `test_fractal_dim_oracle.py` | box-count vs **ImageJ/FracLac** goldens (1.8706, 1.0493) *and* closed-form dimensions for square/line/Sierpinski/Koch | `test_morphology_fraclac.py`, methods split `_fraclac` vs `_analytic` |

Only the third was an oracle, and it is two kinds at once.

**`test_morphology_cellprofiler.h` created** for the 6 rows column J names (the 5 `EDGE_*` edge
intensities and `MASS_DISPLACEMENT`). They were asserted inside two *mixed-kind* functions —
`test_morphology_contour_regression` (5 `EDGE_*` beside `PERIMETER`) and
`test_morphology_basic_regression` (`MASS_DISPLACEMENT` among 14 others) — so the split is **per
feature, not per function**, and the regression functions keep the features that are genuinely theirs.

The shared golden map is still called `unvetted_nyxus_regression_shape2d_feature_golden_values` and
carries no CellProfiler version, config or generator. **The registry is the authority on what vets a
feature; a stale table name is not evidence against it.** The missing SPEC §6.4 provenance record is
tracked in `not_covered.md` §C, and renaming golden tables is deferred to a pass that handles them
across the whole tree.

## 5.34 Wave 24 (glcm) — executed

The largest family: 140 functions, 118 rows, six files. Verified: **gtest 731/731**, 111 glcm cases
(28 ibsi + 10 matlab + 23 pyradiomics + 43 regression + 1 mechanics).

- **`test_glcm_matlab.h` created** for the 10 rows column J names — `ASM`, `CONTRAST`,
  `CORRELATION`, `ENERGY`, `HOM1` and their `_AVE` twins. Each was a one-feature function in
  `test_glcm_regression.h`, so they moved whole. The config is the family default (GREYDEPTH=100,
  offset 1, asymmetric matrix) — the path MATLAB `graycomatrix`/`graycoprops` is matched on, which is
  why these ten and not the IBSI-path features.
- **`test_3d_glcm.h` → `test_3d_glcm_regression.h`**, satisfying column J for 29 3D rows. Still not
  `#include`d, so its 25 assertions remain dead (§B.1).
- **6 rows repointed to `test_glcm_pyradiomics.py`** (`ID`, `IDN`, `IDM`, `IDMN` and two `_AVE`
  twins) — the Python oracle genuinely asserts those against PyRadiomics. The other 28 keep pointing
  at the unwritten `test_glcm_pyradiomics.h`: repointing all 34 would have manufactured a claim for
  28 features the `.py` never touches. **glcm's dangling targets drop 80 → 33.**
- **140 functions renamed**; 4 helpers → `assert_*`.

**A blind spot in the case-name check, found in review.** Six `_AVE` functions
(`test_glcm_DIFAVE_AVE`, `..._DIFENTRO_AVE`, `..._DIFVAR_AVE`, `..._SUMAVERAGE_AVE`,
`..._SUMENTROPY_AVE`, `..._SUMVARIANCE_AVE`) kept their pre-spec mixed-case names with no kind
suffix. They passed the `case == UPPER(function)` check *trivially*, because the function name was
already uppercase - that check verifies function-to-case agreement, not whether the function follows
the convention, so a function that is simply never renamed satisfies it. Renamed to match their own
non-`_AVE` siblings in the same file (`test_glcm_difference_average_ave_regression`, not the
abbreviated `difave`). The checker added in Wave 27 tests the convention directly and would have
caught these.

**A `current_test` reference that was never true.** `test_glcm.h` sat in `current_test` on three
rows and named a file Wave 2 had renamed. The obvious fix is to repoint it - but the three rows are
the *first-order* `ENERGY`, `ENTROPY` and `VARIANCE`, and `test_glcm_regression.h` asserts only
`Feature2D::GLCM_ENERGY`/`GLCM_ENTROPY`/`GLCM_VARIANCE`, never the bare first-order features. The
reference was a name-collision artifact of the original audit scan, so it is **removed**, not
renamed. Tree-wide, `current_test` now has zero references to nonexistent files.

**A fourth case-name defect class:** 8 cases carried a spurious `DIFFERENCE_` prefix —
`TEST_IBSI_GLCM_DIFFERENCE_ID`, `..._IDN`, `..._IDM`, `..._IDMN` and their `COMPAT_3GLCM` twins. ID,
IDN, IDM and IDMN are *inverse-difference* features; the cases named them as difference features,
which are the separate `DIFAVE`/`DIFENTRO`/`DIFVAR` set asserted in the same file. Also
`TEST_IBSI_CONTRAST` was missing its family entirely.

Running tally of case-name defects the `UPPER(function)` rule has surfaced: dropped `M` in three
families, lowercase names, a feature that does not exist (`SALGLZE`), a stray `MATRIX_` infix, and
now a wrong feature-class prefix. None was findable by reading a single file.

## 5.35 Waves 25-26 (hu, plumbing) — executed; the tree now conforms

The last two rename waves, both decision-free: neither family has registry rows, so nothing had to be
placed — only kinds assigned. Verified: **gtest 731/731**, pytest unchanged.

- **hu (19 functions):** `_analytic` for the three closed-form assertions on the uint-friendly HU
  mapping, `_mechanics` for the eight loader tests, `_regression` for the Python offset-domain
  snapshots, `_pydicom` for the two that compare against pydicom-decoded CT values.
  `test_hounsfield{,_nifti}.py` → `test_hu{,_nifti}_regression.py`; the fixture README's references
  followed.
- **plumbing (33 functions):** arrow, omezarr, tiff-loader, initialization, roi-blacklist, 3d-nifti,
  ooc and the vetting self-tests, all `_mechanics` except the nine out-of-core equality checks, which
  are `_invariant`. Six parameterized arrow helpers → `assert_arrow_file_naming_case_N`.
- **The last five drifted case names** were corrected here, including a misspelling that had survived
  since the file was written: `TEST_3D_NIFTY_LOADER` / `TEST_3D_NIFTY_DACC_CONSISTENCY` →
  `TEST_3D_NIFTI_*` (the format is NIfTI, and the *file* was already spelled correctly).
- **The last two non-conforming files:** `test_morphology_features.h` held a single MATLAB-`bwperim`
  PERIMETER test and was folded into `test_morphology_matlab.h`; `test_feature_calculation.h` →
  `test_feature_calculation_common.h` (a shared template helper, asserts nothing itself).

**Tree-wide conformance, measured:**

| | checked | non-conforming |
|---|---:|---:|
| test files (SPEC §6.1) | 107 | **0** |
| test functions (SPEC §6.2) | 668 | **0** |
| gtest cases = `UPPER(function)`, suite `TEST_NYXUS` | 526 (526 unique) | **0** |

One false positive was caught and reverted in the plumbing sweep: `test_nyxus.py` passes the string
`"test_parquet"` as an output *filename*, not a test name. It is the one file the rollout leaves
grandfathered (§4: its 88 API assertions need a by-family `_mechanics` split, not a rename), so it
must not be touched by a name sweep at all.

---

## 6. Reconciliation decisions (RESOLVED)

1. **SPEC §4 oracle-token set** — add **`skimage`** (mainstream; 60+ moment features + circularity).
   `DIAMETER_MIN_ENCLOSING_CIRCLE` vets against **imea** (already a token), not OpenCV. `mahotas`,
   `DIPlib`, `Centrosome` are **not** accepted → features only they cover stay regression/analytic
   (so `ZERNIKE2D` is **analytic**, not mahotas-vetted).
2. **Feature count 758 vs 705** — `featureset.h` is authoritative. The 53-feature delta is fully
   explained: **47 IH `_VAL` variants** (bin-center twins of the `_IDX` features; analytic per §5.8)
   **+ 6 IMQ**. No mystery features; no further research.
3. **Mechanics/fixture files** — rename I/O + plumbing tests to `test_<area>_mechanics.*`
   (`test_tiff_loader`, `test_omezarr`, `test_arrow*`, `test_3d_nifti`, `test_initialization`,
   `test_roi_blacklist`, and the API assertions in `test_nyxus.py`). Leave pure fixtures/harness
   (`test_data.h`, `test_main_nyxus.h`, `test_dsb2018_data.h`, `test_tissuenet_data.py`) **unrenamed**.
4. **IMQ + Gabor naming** — IMQ → `test_imq_<kind>.h` (single prefix, `imq_imq` glitch fixed). Gabor
   keeps its own 1-feature family file (`test_gabor_regression.h`); `ZERNIKE2D` → `test_zernike_regression.h`.
5. **`test_3d_feature_coverage.h` split** — confirmed **in scope** for the 3D waves (213 assertions →
   per-family `test_3d_<family>_*` files); the single biggest mechanical task.

---

## 7. Next steps

- **Phase 2 (discussion):** walk §5 family-by-family, pick oracles for `research-*`, triage the 20
  `suspected-bug` rows, and settle §6. Record decisions in `oracle_coverage.csv` (`oracle`, `status`,
  `notes`) and here.
- **Phase 3 (code):** scaffold (`config_recipes.md`, `check_coverage.py`, `matrix/`, `oracles/gen_*`),
  then GLCM as the worked template, then family-by-family per §3, each wave green.
