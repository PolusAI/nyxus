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
| **glcm** | 118 (72/46/0) | pyradiomics, matlab, mirp | `test_glcm.h`, `test_glcm_oracle.h`, `test_ibsi_glcm.h`, `test_3d_glcm.h`, `test_compat_3d_glcm.h`, `test_glcm_oracle.py` | `test_glcm_pyradiomics.h`, `test_glcm_matlab.h`, `test_glcm_ibsi.h`, `test_glcm_regression.h`, `test_3d_glcm_pyradiomics.h`, `test_3d_glcm_mirp.h`, `test_3d_glcm_regression.h`, `test_glcm_pyradiomics.py` |
| **morphology** | 113 (39/74/0) | matlab, cellprofiler, skimage, imea | `test_shape_morphology_2d.h`, `test_2d_remaining_features.h`, `test_3d_shape.h`, `test_feature_oracle.py`, `test_convex_hull_invariants.py`, `test_fractal_dim_oracle.py` | `test_morphology_matlab.h`, `test_morphology_cellprofiler.h`, `test_morphology_skimage.h`, `test_morphology_imea.h`, `test_morphology_regression.h`, `test_morphology_invariant.py`, `test_3d_morphology_*.h` |
| **firstorder** | 72 (51/21/0) | matlab, pyradiomics | `test_pixel_intensity_features.h`, `test_ibsi_intensity.h`, `test_3d_inten.h`, `test_compat_3d_fo_radiomics.h` | `test_firstorder_matlab.h`, `test_firstorder_pyradiomics.h`, `test_firstorder_regression.h`, `test_3d_firstorder_pyradiomics.h`, `test_3d_firstorder_regression.h` |
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
- Harness/fixtures (`test_main_nyxus.h`, `test_data.h`, `test_dsb2018_data.h`, `test_tissuenet_data.py`) and pure I/O mechanics (`test_tiff_loader.*`, `test_omezarr.h`, `test_arrow*.h`, `test_3d_nifti.h`, `test_initialization.h`, `test_roi_blacklist.h`, `test_feature_calculation.h`) are **out of the family taxonomy** — keep as-is or move under a `_mechanics` convention (decision below).

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

## 6. Open reconciliation decisions (need a call before Phase 3)

1. **SPEC §4 oracle-token set is incomplete.** Add **`skimage`** (accepted), and decide on tokens for
   the non-mainstream tools seen in the tracker (`diplib`/`mahotas`/`centrosome`) — currently routed to
   regression. Recommend: extend §4 to list `skimage`; leave the others unlisted (regression).
2. **Feature count 758 (featureset.h) vs 705 (tracker).** Confirm `featureset.h` is authoritative; the
   53 `not-in-tracker` rows are the delta to classify.
3. **Mechanics/fixture files** (§4 list) — do they get a `_mechanics` suffix and stay in `tests/`, or
   stay untouched? Recommend: rename the plumbing tests to `test_<area>_mechanics.*`; leave pure
   fixtures/harness (`test_data.h`, `test_main_nyxus.h`) unrenamed.
4. **IMQ + Gabor naming.** IMQ → `test_imq_*` (no double prefix); Gabor is a 1-feature family — confirm
   `test_gabor_*` vs folding into morphology/texture.
5. **`test_3d_feature_coverage.h` split** — the single biggest mechanical task (213 assertions → per-family 3D files). Confirm this split is in scope for the 3D waves.

---

## 7. Next steps

- **Phase 2 (discussion):** walk §5 family-by-family, pick oracles for `research-*`, triage the 20
  `suspected-bug` rows, and settle §6. Record decisions in `oracle_coverage.csv` (`oracle`, `status`,
  `notes`) and here.
- **Phase 3 (code):** scaffold (`config_recipes.md`, `check_coverage.py`, `matrix/`, `oracles/gen_*`),
  then GLCM as the worked template, then family-by-family per §3, each wave green.
