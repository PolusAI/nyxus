# Nyxus Test-Suite Vetting Audit

**Goal:** understand what the test suite currently *vets against an oracle*, what is only
*regression-guarded* (self-snapshot), and what is *untested* — so we can drive the un-vetted
features toward validation with existing oracles (IBSI, pyradiomics, scikit-image, analytic).

**Scope:** all **758** real, compiled features in `src/nyx/featureset.h`
(`Feature2D` + `Feature3D` + `FeatureIMQ`; group/family markers and `_FIRST/_LAST/ALL_*`
sentinels excluded).

> **Structural correction (2026-07-11):** the `Feature3D` enum contains a 31-entry
> `#if 0 // 3D features planned for a future PR` block — 9 centroid/bbox, 9 neighbor, and
> 13 spatial-moment features that are **compiled out**: not selectable, not computed, not
> vettable. An earlier pass counted them (789 total, 52 untested). They are excluded here,
> giving **758 real features** and **21 genuinely untested**. The 21 are *all* implemented
> 2D intensity-histogram variants — every previously-flagged "untested 3D" feature was one of
> these phantom placeholders.

**Method:** every feature enum was cross-referenced against every test file two ways —
(1) the four-agent per-feature audit that read each test's *assertions* and named the oracle,
and (2) an authoritative token scan (`scan_tests.py`) that word-boundary + dimension-aware
matches each enum name across all 65 test files. The **strongest** evidence tier wins per
feature. The scan corrected ~45 agent false-negatives (the agents sampled base GLCM/GLRLM
names and skipped the `_AVE` variants, which *are* regression-tested).

---

## Status taxonomy

| Status | Meaning | Trust |
|---|---|---|
| **VETTED** | Compared to an external reference (IBSI / pyradiomics / scikit-image / ImageJ / MATLAB) or a closed-form analytic value | High — real V&V |
| **VETTED-CLAIMED** | Test uses `assert_verifiable_with_3p_builtin_oracle_*`, but the oracle tool is **not named in the code** — the golden number's provenance can't be confirmed from the repo | Medium — a documentation gap, not necessarily wrong |
| **REGRESSION-ONLY** | Golden value is a self-snapshot of a past Nyxus run — guards against drift, does **not** confirm correctness | Low — no independent truth |
| **NOT TESTED** | No test asserts a value for this feature | None |

---

## Headline numbers (758 features)

| Status | Count | % |
|---|---:|---:|
| VETTED (oracle / analytic) | **331** | 44% |
| VETTED-CLAIMED (oracle not named) | **121** | 16% |
| REGRESSION-ONLY (self-snapshot) | **285** | 38% |
| NOT TESTED | **21** | 3% |

**Only 44% of features are vetted against a nameable oracle today.** 737/758 features have at
least one value-checking test, but for 406 of them (54% of the suite) that check is either a
self-snapshot or an unnamed "builtin oracle." The 21 untested are all 2D intensity-histogram
dispersion/index variants (see Gap 2).

### By feature group

| group | VETTED | CLAIMED-3P | REGRESSION | NOT-TESTED | total |
|---|---:|---:|---:|---:|---:|
| texture (GLCM/GLRLM/GLSZM/GLDM/GLDZM/NGTDM/NGLDM/Gabor) | 233 | 0 | 94 | 0 | 327 |
| morphology-extended (caliper/chord/convex/Zernike…) | 35 | 30 | 22 | 0 | 87 |
| moments/histogram (spatial/central/normalized/Hu moments) | 0 | 86 | 94 | 0 | 180 |
| intensity (first-order) | 32 | 4 | 34 | 0 | 70 |
| morphology (area/perimeter/centroid/bbox/axes…) | 5 | 1 | 35 | 0 | 41 |
| intensity-histogram | 26 | 0 | 0 | 21 | 47 |
| image-quality (IMQ) | 0 | 0 | 6 | 0 | 6 |

*(The 3D centroid/bbox, neighbor, and spatial-moment rows that previously appeared under
morphology / morphology-extended / moments as "NOT-TESTED" are gone — they were the `#if 0`
placeholders.)*

### VETTED — by oracle actually used (feature-token counts)

- **pyradiomics** — 212 (2D GLCM/GLRLM/GLDM oracle tests + all 3D `test_compat_3d_*`)
- **IBSI** — 251 (2D texture `test_ibsi_*` + `test_ibsi_intensity`)
- **analytic** — 58 (intensity-histogram closed forms, fractal analytic, chord angles, ASM/contrast identities)
- **scikit-image / ImageJ / MATLAB** — a handful (convex-hull invariants, fractal box-count, 3D covariance)

Texture is the strong point: **IBSI-vetted in 2D, pyradiomics-vetted in 3D.**

---

## The gaps, in priority order

### GAP 1 — Intensity-histogram dispersion/index variants (21 NOT TESTED — the *only* untested implemented features)

`test_intensity_histogram.*` vets the core histogram stats analytically, but the derived
dispersion and per-bin (`_IDX`) variants have **no test at all**:

- Dispersion `_VAL` (7): IQR, MAD, robust-MAD, median-AD, CoV, quantile-coeff-of-dispersion, robust-mean.
- The entire `_IDX` family (14): variance, skewness, kurtosis, IQR, range, MAD×3, CoV, QCoD,
  entropy, uniformity, robust-mean — reported in *bin-index* space.
- **How to vet:** the `_VAL` forms map to IBSI first-order / pyradiomics `firstorder`
  (MAD, robust-MAD, IQR, entropy, uniformity are all IBSI-defined). CoV, QCoD, and the `_IDX`
  forms are Nyxus-specific (no standard-tool equivalent) → verify analytically on a tiny fixture.

### GAP 2 — Moments: 180 features, 0 vetted against a nameable oracle

Every spatial/central/normalized/weighted/Hu moment (shape **and** intensity, 2D) is either
**CLAIMED-3P** (86: labeled "builtin oracle" in `test_2d_geometric_moments.h`, but no tool
named) or **REGRESSION-ONLY** (94: weighted/normalized-raw moments).

- Raw/central/normalized/Hu moments are exactly
  `skimage.measure.moments / moments_central / moments_normalized / moments_hu`. One parametrized
  test would convert ~86 CLAIMED + a chunk of REGRESSION to genuinely **VETTED** and confirm
  whether the existing "builtin" goldens are correct.
- **Note:** the 3D `SPAT_MOMENT_*` are **not** in scope — they are `#if 0` (unimplemented).

### GAP 3 — Regression-only clusters (real, but self-snapshot only)

- **2D basic morphology (35 regression):** area, perimeter, centroid, bbox, axes, eccentricity,
  extent, orientation, Euler, equivalent diameter — directly covered by `skimage.regionprops`.
  Only 5 are vetted today.
- **3D texture NGLDM (19) + GLDZM (18):** regression-only because pyradiomics has **no**
  NGLDM/GLDZM — need the **IBSI 3D digital phantom** reference values instead.
- **Image-quality (6):** focus, local-focus, power-spectrum slope, saturation, sharpness —
  no external oracle wired up; compare to a reference implementation or note as regression.

### NOT A GAP — the 31 `#if 0` 3D features

3D `CENTROID_{X,Y,Z}`, `BBOX_*`, all neighbor features, and 3D `SPAT_MOMENT_*` are behind
`#if 0 // 3D features planned for a future PR`. They cannot be tested or vetted until implemented.
**These are a feature-implementation task, not a vetting task.**

---

## Recommended roadmap (biggest oracle-coverage gain first)

| # | Action | Converts | Oracle | Effort |
|---|---|---|---|---|
| 1 | Intensity-histogram `_VAL`/`_IDX` variants — `_VAL` vs IBSI/pyradiomics, `_IDX`+CoV+QCoD analytically | 21 NOT-TESTED | IBSI / pyradiomics / analytic | Low–Medium |
| 2 | Parametrized 2D moments test vs scikit-image (shape + intensity; raw/central/normalized/Hu) | ~86 CLAIMED + chunk of REGRESSION | `skimage.measure.moments*` | Medium |
| 3 | 2D basic-morphology test vs `skimage.regionprops` (area, perimeter, centroid, bbox, axes, eccentricity, extent, orientation, Euler, eq-diameter) | ~35 REGRESSION | scikit-image | Medium |
| 4 | 3D NGLDM + GLDZM vs IBSI 3D digital phantom reference | 37 REGRESSION | IBSI | Medium–High |
| 5 | **Name the oracle** for the existing "builtin" moment/caliper/chord goldens (or downgrade to regression) | 121 CLAIMED → honest label | doc-only | Low |
| 6 | Image-quality features vs a reference focus/sharpness implementation | 6 REGRESSION | reference impl | Medium |
| — | *(separate track)* implement the 31 `#if 0` 3D geometry/neighbor/moment features, then vet | 31 not-implemented | skimage/pyradiomics | Large |

**Note on the CLAIMED-3P bucket (121):** these tests *look* vetted
(`assert_verifiable_with_3p_builtin_oracle_*`) but the code never records *which* third-party
tool produced the golden number. Actions #2 and #5 close that provenance gap.

---

## Files in this folder

Tracked: `vetting_report.csv` (**master** — all 758 features: `dim, group, family, feature,
vetting_status, oracle_used, test_files, candidate_oracle, how_to_vet`) and this report.

Regenerable byproducts (git-ignored; rebuilt by `report.py`): `vetting_pivot.csv` (group×status —
also shown as a table above), `gap_not_tested.csv` (21 untested), `gap_claimed_3p.csv` (121
oracle-not-named), `gap_regression.csv` (285 self-snapshot-only), plus `features.csv` /
`audit_scan.txt` intermediates.

Regenerate: `extract_features.py` → `scan_tests.py` → `merge.py` → `report.py` (scripts alongside).
