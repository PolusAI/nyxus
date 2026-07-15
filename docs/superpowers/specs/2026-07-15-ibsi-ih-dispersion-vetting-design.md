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
  - **18 rows → `oracle = ibsi`**: the 12 IDX with published IBSI consensus values, plus the 6
    `_VAL` transform-anchored to them (IQR, MAD, rMAD, medianAD, CoV, QCoD). `_VAL` notes document
    the affine transform (`VAL = binWidth·IDX`, etc.) and the analytic robust-fixture corroboration.
  - **2 rows → `oracle = analytic`**: `ROBUST_MEAN_IDX` and `ROBUST_MEAN_VAL` — robust mean is not an
    IBSI feature, so these are vetted on the discriminating fixture (hand-computed) and via the
    VAL/IDX relation, with notes stating no IBSI equivalent exists.
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
