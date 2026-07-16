# IBSI-vetted Intensity-Histogram dispersion/index tests — design

**Date:** 2026-07-15
**Status:** Approved (brainstorming) — pending spec review, then implementation plan.
**Branch:** `test/vet-ih-dispersion-ibsi` (off `main`).

## 1. Goal

Give **all 20** Intensity-Histogram (IH) dispersion/index features a genuine oracle-vetted
assertion rooted in the **IBSI** reference tables. These 20 features are currently `untested` in
`tests/vetting/oracle_coverage.csv`:

- **13 `_IDX` variants:** `IH_VARIANCE_IDX`, `IH_SKEWNESS_IDX`, `IH_EXCESS_KURTOSIS_IDX`,
  `IH_INTERQUANTILE_RANGE_IDX`, `IH_RANGE_IDX`, `IH_MEAN_ABSOLUTE_DEVIATION_IDX`,
  `IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX`, `IH_MEDIAN_ABSOLUTE_DEVIATION_IDX`,
  `IH_COEFFICIENT_OF_VARIATION_IDX`, `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX`,
  `IH_ENTROPY_IDX`, `IH_UNIFORMITY_IDX`, `IH_ROBUST_MEAN_IDX`.
- **7 `_VAL` variants:** the `_VAL` counterparts of `INTERQUANTILE_RANGE`, `MEAN_ABSOLUTE_DEVIATION`,
  `ROBUST_MEAN_ABSOLUTE_DEVIATION`, `MEDIAN_ABSOLUTE_DEVIATION`, `COEFFICIENT_OF_VARIATION`,
  `QUANTILE_COEFFICIENT_OF_DISPERSION`, `ROBUST_MEAN`.

## 2. Background & decision

- PR 367 vetted these 20 analytically on tiny hand-computed fixtures. **367 will be closed**; this
  branch replaces it and becomes the sole vetting for the 20 features.
- **IBSI is the correct external oracle for the `_IDX` family:** IBSI computes intensity-histogram
  statistics over the discretised grey-level *indices*, which is exactly Nyxus's IDX domain. This
  gives the Nyxus-specific IDX features real external validation (the gap 367's analytic-only
  approach left open).
- **The `_VAL` features have no direct external oracle** (they use bin-*center* values — a
  Nyxus-specific representation; no mainstream tool computes bin-center dispersion). pyradiomics is a
  value-domain firstorder tool that maps to Nyxus's separate FirstOrder family, and it lacks
  MedianAD / CoV / QCoD / RobustMean entirely, so it can reach only `ENTROPY_IDX`/`UNIFORMITY_IDX`.
- **`_VAL` is a deterministic transform of `_IDX`.** In `intensity_histogram.cpp`,
  `binCenter(i) = minVal + (i + 0.5)·binWidth`, and `_VAL`/`_IDX` are the same statistic computed in
  parallel over `binCenter(i)` vs `i`. Since value is affine in index, every `_VAL` is a closed-form
  transform of the IDX distribution, so it is **transitively anchored to the IBSI-vetted IDX
  values** (see §5).

## 3. Oracle & fixtures

**Primary fixture — IBSI digital phantom.** Reuse the in-repo phantom (`ibsi_phantom_z1..z4_intensity`
+ masks) and the multi-slice assembly pattern from `test_firstorder_ibsi.h`. Assert the 13 IDX
features against IBSI **intensity-histogram** consensus values (family 3.4 — distinct from the
intensity-based/first-order family 3.3 that `test_firstorder_ibsi.h` already covers).

**Secondary fixture — robust-window discriminating ROI (analytic).** Carry over a small
hand-computed fixture (in the spirit of 367's 17-px ROI, `freq = {1,5,6,4,1}`) whose robust window
`[p10Idx, p90Idx]` strictly trims the tail bins, so `robust ≠ full`. This preserves coverage of the
**robust-window trimming path** (`ROBUST_MEAN_*`, `ROBUST_MEAN_ABSOLUTE_DEVIATION_*`), which the IBSI
phantom may not exercise. Its goldens are hand-derived (analytic), recorded as corroboration.

*Implementation note:* confirm empirically whether the IBSI phantom's IH robust window already trims.
If it does, the discriminating fixture is redundant for the robust path and can be dropped; if not (the
likely case), it is required.

## 4. IDX vetting (direct, against IBSI)

Compute the IH family on the phantom under the IBSI-matching config and assert the 13 IDX features
against the IBSI IH consensus table.

- 12 of 13 have published IBSI IH consensus values. **`ROBUST_MEAN_IDX` has no IBSI feature**
  (IBSI publishes robust-MAD but not a standalone robust mean) → it is vetted on the discriminating
  fixture (analytic) and via the VAL/IDX relation, not against an IBSI number.

## 5. VAL anchoring (transform of the IBSI-vetted IDX)

With `a = minVal + 0.5·binWidth`, `b = binWidth`, and `deltaValue = b·deltaIndex` exactly:

| `_VAL` feature | relation to the IDX quantity |
|---|---|
| `INTERQUANTILE_RANGE_VAL` | `b · IQR_IDX` |
| `MEAN_ABSOLUTE_DEVIATION_VAL` | `b · MAD_IDX` |
| `ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL` | `b · rMAD_IDX` |
| `MEDIAN_ABSOLUTE_DEVIATION_VAL` | `b · medAD_IDX` |
| `ROBUST_MEAN_VAL` | `a + b · rMean_IDX` |
| `COEFFICIENT_OF_VARIATION_VAL` | `(b·σ_IDX) / (a + b·mean_IDX)` — from IDX moments + (minVal, binWidth) |
| `QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` | `(Q3−Q1)/(Q3+Q1)` in value domain — from IDX quantiles + (minVal, binWidth) |

Each `_VAL` is asserted against the value obtained by transforming the **IBSI-anchored IDX**
quantity with the run's `(minVal, binWidth)` — so correctness rests on the IBSI consensus plus
documented arithmetic, not on self-consistency alone. `minVal`/`binWidth` are read from the same run
(`IH_MINIMUM_VAL`, `IH_BIN_SIZE`) and cross-checked.

## 6. Central technical risk — config & index-convention matching

This is where the work concentrates and must be resolved before any golden is pinned:

1. **Discretisation config.** The phantom's IH benchmark uses a specific bin width / bin count.
   Nyxus's `ih_make_settings` (GREYDEPTH / IBSI knobs) must be set to match it. Documented as a new
   `ih.ibsi_fbn` config recipe.
2. **Index base offset.** IBSI grey levels are 1-based; Nyxus IDX is 0-based (see
   `intensity_histogram.cpp` — index `i` runs `0..N-1`). Shift-invariant features (variance,
   skewness, excess-kurtosis, IQR, range, MAD, rMAD, medianAD, entropy, uniformity) match regardless.
   **Shift-sensitive features (`CoV`, `QCoD`, and any location term) carry the offset** and must be
   reconciled explicitly. The exact convention will be determined empirically (run Nyxus on the
   phantom, compare to the IBSI table) before pinning values.

### 6.3 Resolution — Task 1 characterisation spike (2026-07-16)

The two unknowns are now resolved empirically. A throwaway dump harness ran the IH family on the
masked digital phantom (4 slices concatenated — IH depends only on the intensity multiset, so the
3D-volume aggregation IBSI recommends (DHQ4) is reproduced by concatenation) and the output was
compared against the sourced IBSI table below.

**Config — `IH_PHANTOM_NBINS = 6`.** Setting `GREYDEPTH = 6` with `IBSI = true` reproduces the IBSI
digital-phantom IH benchmark. This is FBN=6, exactly the discretisation the IBSI reference data-set
page prescribes for the phantom ("fixed bin number discretisation of 6 bins"; equivalently FBS=1 or
"as is"). Under FBN=6 the phantom's discretised grey levels come out `{1,3,4,6}` — identical to the
original integer intensities — which is why the IH values equal the intensity-based (first-order)
values. `GREYDEPTH = 1` is rejected by Nyxus and returns the soft-NaN sentinel (-7777) for the whole
family, so it is not a usable config.

**Index base — `IH_PHANTOM_INDEX_BASE = 1` (matches IBSI directly, NO offset correction).** The
earlier 0-based speculation in §6.2 was wrong. Empirically Nyxus reports IDX in the **1-based**
grey-level convention: `IH_MEAN_IDX = 2.149 ≈ 2.15` (the IBSI 1-based mean, not 1.15). Decisively,
the two shift-sensitive features also match IBSI with no correction: `IH_COEFFICIENT_OF_VARIATION_IDX
= 0.8122 ≈ 0.812` and `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX = 0.600 ≈ 0.6`. Had Nyxus been
0-based, CoV would be ~1.518 and QCoD ~1.0. Consequence: Task 2 asserts **all** IDX features
(shift-invariant AND shift-sensitive) directly against the IBSI consensus values — no index-base
arithmetic is needed. (This agrees with the `intensity_histogram.h` header comment "...Index =
1-based bin index" and the existing regression test's `IH_MEAN_IDX = 2.0` expectation.)

**Reconciliation evidence (Nyxus IDX @ GREYDEPTH=6, IBSI mode — vs IBSI consensus):**

| Feature | IBSI dig.phantom | Nyxus IDX | rel. err |
|---|---|---|---|
| variance | 3.05 | 3.04547 | 0.15% |
| skewness | 1.08 | 1.08382 | 0.35% |
| excess kurtosis | −0.355 | −0.35462 | 0.11% |
| interquartile range | 3 | 3 | 0 |
| range | 5 | 5 | 0 |
| mean abs. deviation | 1.55 | 1.55223 | 0.14% |
| robust mean abs. dev. | 1.11 | 1.11383 | 0.35% |
| median abs. deviation | 1.15 | 1.14865 | 0.12% |
| coefficient of variation | 0.812 | 0.812198 | 0.02% |
| quartile coeff. dispersion | 0.6 | 0.600 | 0 |
| entropy | 1.27 | 1.26561 | 0.35% |
| uniformity | 0.512 | 0.512418 | 0.08% |
| mean (loc.) | 2.15 | 2.14865 | 0.06% |
| median / min / P10 / mode | 1 | 1 | 0 |
| P90 | 4 | 4 | 0 |
| maximum | 6 | 6 | 0 |
| max hist. gradient | 8 | 8 | 0 |
| max hist. gradient intensity | 3 | 3 | 0 |
| min hist. gradient | −50 | −50 | 0 |
| min hist. gradient intensity | 1 | 1 | 0 |

Every feature agrees within rel 1e-2 (all ≤ 0.35%), including the shift-sensitive CoV/QCoD. No `src/`
change was required. The scratch harness was removed; nothing test-code was committed.

`_VAL` features (bin-center domain) are, as designed, a distinct Nyxus-specific representation with no
IBSI oracle — e.g. `IH_MEAN_VAL = 2.374` (bin centers `minVal + (i+0.5)·binWidth`, binWidth = 5/6 =
0.8333, minVal = 1). They are anchored transitively per §5, not against these IBSI numbers.

### 6.4 Sourced IBSI IH goldens + provenance (for Task 2's `ibsi_ih_phantom_golden`)

**Source (authoritative oracle):** Zwanenburg et al., *The Image Biomarker Standardisation
Initiative*, arXiv:1612.07003 — Chapter 3, §3.4 "Intensity histogram features", the per-feature
"Reference values" tables, **`dig. phantom` column**. Config: digital phantom, FBN = 6 bins
(equivalently FBS = 1 / no discretisation), 3D-volume aggregation. Cross-checked against
ibsi.readthedocs.io `03_Image_features.html` and the reference-data-sets page (`05_Reference_data_sets.html`,
which states the phantom's IH discretisation). Overlapping values (mean, variance, skewness, kurtosis,
median, IQR, range, MAD, rMAD) additionally corroborate the repo's already-vetted first-order goldens
in `tests/test_firstorder_ibsi.h`.

| IBSI feature (code) | Table | dig. phantom value | consensus |
|---|---|---|---|
| Mean discretised intensity (X6K6) | 3.47 | 2.15 | very strong |
| Discretised intensity variance (CH89) | 3.48 | 3.05 | strong |
| Discretised intensity skewness (88K1) | 3.49 | 1.08 | very strong |
| (Excess) discretised intensity kurtosis (C3I7) | 3.50 | −0.355 | very strong |
| Median discretised intensity (WIFQ) | 3.51 | 1 | very strong |
| Minimum discretised intensity (1PR8) | 3.52 | 1 | very strong |
| 10th discretised intensity percentile (GPMT) | 3.53 | 1 | very strong |
| 90th discretised intensity percentile (OZ0C) | 3.54 | 4 (some impls give 4.2) | very strong |
| Maximum discretised intensity (3NCY) | 3.55 | 6 | very strong |
| Intensity histogram mode (AMMC) | 3.56 | 1 | very strong |
| Discretised intensity interquartile range (WR0O) | 3.57 | 3 | very strong |
| Discretised intensity range (5Z3W) | 3.58 | 5 | very strong |
| IH mean absolute deviation (D2ZX) | 3.59 | 1.55 | very strong |
| IH robust mean absolute deviation (WRZB) | 3.60 | 1.11 | very strong |
| IH median absolute deviation (4RNL) | 3.61 | 1.15 | very strong |
| IH coefficient of variation (CWYJ) | 3.62 | 0.812 | very strong |
| IH quartile coefficient of dispersion (SLWD) | 3.63 | 0.6 | very strong |
| Discretised intensity entropy (TLU2) | 3.64 | 1.27 | very strong |
| Discretised intensity uniformity (BJ5W) | 3.65 | 0.512 | very strong |
| Maximum histogram gradient (12CE) | 3.66 | 8 | very strong |
| Maximum histogram gradient intensity (8E6O) | 3.67 | 3 | strong |
| Minimum histogram gradient (VQB3) | 3.68 | −50 | very strong |
| Minimum histogram gradient intensity (RHQZ) | 3.69 | 1 | strong |

The 12 Task-scope dispersion/index goldens Task 2 must pin (all against the IDX variant, index base 1):
variance 3.05, skewness 1.08, excess kurtosis −0.355, interquartile range 3, range 5, mean absolute
deviation 1.55, robust mean absolute deviation 1.11, median absolute deviation 1.15, coefficient of
variation 0.812, quartile coefficient of dispersion 0.6, entropy 1.27, uniformity 0.512.
`ROBUST_MEAN_IDX` has no IBSI feature (IBSI publishes robust-MAD but not a standalone robust mean) →
per §4 it is vetted analytically, not against an IBSI number.

## 7. Test structure (taxonomy, SPEC §6)

- New oracle file **`tests/test_intensity_histogram_ibsi.h`**.
- Function **`test_ih_dispersion_ibsi`** (IDX-vs-IBSI + VAL-transform asserts on the phantom).
- Function **`test_ih_dispersion_robust_analytic`** (discriminating-fixture robust-window checks).
- Register both in `tests/test_all.cc`.
- Phantom data reused from `test_data.h` / the existing IBSI test; IH helpers reused from
  `test_intensity_histogram_regression.h` where shareable (else duplicated minimally in the new file).

## 8. Registry & config-recipe updates

- Add `ih.ibsi_fbn` to `tests/vetting/config_recipes.md` (bin config + oracle = ibsi; note the
  IDX↔IBSI index-domain correspondence and the VAL = transform(IDX) relation).
- Flip the rows `untested → vetted`, `config_recipe = ih.ibsi_fbn`,
  `current_test = test_intensity_histogram_ibsi.h`, with oracle tokens:
  - **16 rows → `oracle = ibsi`**: the 12 IDX with published IBSI consensus values, plus the 4
    cleanly IBSI-anchorable `_VAL` — MAD, rMAD, medianAD (each `= binWidth·IDX`) and CoV
    (`= binWidth·sqrt(VARIANCE_IDX)/MEAN_VAL`, anchored on the IBSI-vetted `VARIANCE_IDX`). `_VAL`
    notes document the transform.
  - **4 rows → `oracle = analytic`**: `ROBUST_MEAN_IDX`, `ROBUST_MEAN_VAL` (robust mean is not an
    IBSI feature), `QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` (needs the unexposed `P25/P75` sum), and
    `INTERQUANTILE_RANGE_VAL` (`IQR_IDX` is bin-floored while `IQR_VAL` is interpolated, so
    `IQR_VAL ≠ binWidth·IQR_IDX`). Vetted on the hand-computed discriminating fixture; notes state
    why no IBSI anchor applies.

  *(Refinements discovered during planning/execution: design first said 18 ibsi + 2 analytic → 17 + 3
  (planning: `QCoD_VAL` shift-sensitive) → 16 + 4 (Task 3 execution: `IQR_VAL` fails the transform
  because `IQR_IDX` floors via `getIndexOf()`; the flooring inconsistency is filed as a Nyxus bug in
  `docs/known-issues/`).)*
- Regenerate `coverage_report.md` via `check_coverage.py --write`. `--check` stays green (`ibsi`
  is an allowed oracle token).

## 9. Provenance (SPEC §6.4)

Goldens are the IBSI intensity-histogram digital-phantom consensus values from the IBSI reference
(Zwanenburg et al., *The Image Biomarker Standardisation Initiative*, 2020 — digital-phantom
benchmark, intensity-histogram family). The exact table + configuration are recorded at the golden
definition site. Sourcing the exact numbers and confirming the phantom IH config is the primary
implementation input.

## 10. Verification

- Build `build_gtest`; `TEST_NYXUS.TEST_IH_DISPERSION_IBSI` and the robust-analytic test pass →
  Nyxus IDX features match IBSI consensus and VAL features match the IBSI-anchored transforms.
- `python3 tests/vetting/check_coverage.py --check` exits 0.
- No feature-count regression vs the pre-branch suite.

## 11. Non-goals

- No production/`src` changes — test + registry only.
- No IH file split into `_common.h`/`_analytic.h` (deferred family-wide refactor, tracked separately).
- The 7 `_VAL` features get no *direct* external tool oracle (none exists); transform-anchoring +
  the analytic robust fixture is the accepted approach.
- No pyradiomics work here (optional future second oracle on `ENTROPY_IDX`/`UNIFORMITY_IDX` only).
