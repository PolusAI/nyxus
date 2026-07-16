# IBSI-vetted IH dispersion/index tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vet all 20 Intensity-Histogram dispersion/index features against IBSI (13 `_IDX` directly vs IBSI intensity-histogram digital-phantom consensus; 7 `_VAL` via the exact affine `VAL = transform(IDX)` relation), plus an analytic discriminating fixture that exercises the robust-window trimming path.

**Architecture:** A new oracle test file `tests/test_intensity_histogram_ibsi.h` runs `IntensityHistogramFeatures` on the in-repo IBSI digital phantom under an IBSI-matching discretisation, asserts IDX features against hardcoded IBSI consensus goldens, and asserts each VAL feature against its IDX quantity transformed with the run's `(minVal, binWidth)`. A second fixture (hand-computed) drives the robust window so `robust ≠ full`. The coverage registry is updated to mark the 20 features vetted.

**Tech Stack:** C++17 GoogleTest (header-per-family, included into `tests/test_all.cc`); Nyxus `IntensityHistogramFeatures`; Python stdlib (`tests/vetting/check_coverage.py`) for the registry; conda `build_gtest` target to build/run.

## Global Constraints

- **Test-only. No `src/` changes.** If a golden disagrees with Nyxus, triage config first; a genuine Nyxus discrepancy is reported, not silently patched here.
- **Reference values are never a CI runtime dependency** — IBSI goldens are hardcoded literals with provenance (SPEC §6.4); CI only builds Nyxus and compares.
- **Taxonomy (SPEC §6.1/§6.2):** file `test_intensity_histogram_ibsi.h`; functions `test_ih_dispersion_ibsi`, `test_ih_dispersion_robust_analytic`; gtest macro name = uppercased function under `TEST_NYXUS`.
- **Oracle tokens** must be in the allowed set (`ibsi`, `analytic` are both allowed). `status=vetted` requires a non-empty `oracle`.
- **Tolerance helper:** `agrees_gt(fval, gt, frac_tolerance)` checks `|fval-gt| <= |gt|/frac_tolerance` (higher = tighter; `1e4` ≈ rel 1e-4). Exact zeros use `ASSERT_NEAR(v, 0.0, 1e-9)` (agrees_gt cannot check gt==0).
- **Feature enum:** `Nyxus::Feature2D::IH_*` (exact tokens verified — e.g. `IH_VARIANCE_IDX`, `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL`, `IH_ROBUST_MEAN_IDX`, `IH_BIN_SIZE`, `IH_MINIMUM_VAL`).
- **Target feature set (20):** IDX (13) = VARIANCE, SKEWNESS, EXCESS_KURTOSIS, INTERQUANTILE_RANGE, RANGE, MEAN_ABSOLUTE_DEVIATION, ROBUST_MEAN_ABSOLUTE_DEVIATION, MEDIAN_ABSOLUTE_DEVIATION, COEFFICIENT_OF_VARIATION, QUANTILE_COEFFICIENT_OF_DISPERSION, ENTROPY, UNIFORMITY, ROBUST_MEAN — each `IH_<name>_IDX`. VAL (7) = the `_VAL` counterparts of INTERQUANTILE_RANGE, MEAN_ABSOLUTE_DEVIATION, ROBUST_MEAN_ABSOLUTE_DEVIATION, MEDIAN_ABSOLUTE_DEVIATION, COEFFICIENT_OF_VARIATION, QUANTILE_COEFFICIENT_OF_DISPERSION, ROBUST_MEAN.

## File Structure

- **Create** `tests/test_intensity_histogram_ibsi.h` — IBSI IH oracle tests + the IBSI goldens map + the analytic robust fixture. One responsibility: IBSI/analytic vetting of the IH dispersion/index family.
- **Modify** `tests/test_all.cc` — `#include` the new header; register two `TEST(TEST_NYXUS, …)` cases.
- **Modify** `tests/vetting/config_recipes.md` — add the `ih.ibsi_fbn` recipe.
- **Modify** `tests/vetting/oracle_coverage.csv` — flip 20 rows to vetted.
- **Regenerate** `tests/vetting/coverage_report.md` — via `check_coverage.py --write`.

Phantom data is reused from `tests/test_data.h` (`ibsi_phantom_z1..z4_intensity` / `_mask`); the masked-ROI loader `load_masked_test_roi_data` and `agrees_gt` come from `tests/test_main_nyxus.h`.

---

### Task 1: Characterise & reconcile — pin the IBSI goldens and the config

**Purpose:** Resolve the central risk (discretisation config + 0-based-Nyxus vs 1-based-IBSI index offset) and produce the **independent** IBSI goldens before any test asserts them. The goldens are IBSI's published digital-phantom intensity-histogram consensus values (family 3.4) — sourced, not derived from Nyxus.

**Files:**
- Create (scratch, not committed): `tests/scratch_ih_dump.h` + a temporary `TEST` to dump actual Nyxus phantom IH outputs.
- Produce (committed at end): the `ibsi_ih_phantom_golden` map inside `tests/test_intensity_histogram_ibsi.h` (created here, filled in Task 2).

**Interfaces:**
- Produces: `IH_PHANTOM_NBINS` (int discretisation setting), `IH_PHANTOM_INDEX_BASE` (0 or 1 — how Nyxus IDX relates to IBSI grey-level index), and `ibsi_ih_phantom_golden` = `std::unordered_map<std::string,double>` keyed by IBSI feature name → consensus value, with a provenance comment block. Consumed by Tasks 2–3.

- [ ] **Step 1: Obtain the IBSI IH consensus values (independent oracle).**
  Source the digital-phantom **intensity-histogram** family consensus values from the IBSI reference (Zwanenburg et al. 2020, digital-phantom benchmark table, IH family) for: variance, skewness, (excess) kurtosis, interquartile range, range, mean absolute deviation, robust mean absolute deviation, median absolute deviation, coefficient of variation, quartile coefficient of dispersion, entropy, uniformity. If the user has these values, use them; otherwise take them from the reference table and record the exact table/config. Record the discretisation the table uses (bin width / count) and the aggregation (2D-per-slice vs 3D). **Do not compute them from Nyxus.**

- [ ] **Step 2: Write a throwaway dump harness to capture Nyxus actuals.**
  In `tests/scratch_ih_dump.h`, load the masked phantom and print every IH feature (IDX + VAL) plus `IH_BIN_SIZE`/`IH_MINIMUM_VAL`, mirroring the assembly in `test_firstorder_ibsi.h`:

```cpp
#pragma once
#include <gtest/gtest.h>
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "test_data.h"
#include "test_main_nyxus.h"

static void ih_dump_phantom(int nbins) {
    std::vector<NyxusPixel> img, msk;
    for (auto* z : {ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                    ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity})
        for (size_t i = 0; i < 20; i++) img.push_back(z[i]);
    for (auto* z : {ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                    ibsi_phantom_z3_mask, ibsi_phantom_z4_mask})
        for (size_t i = 0; i < 20; i++) msk.push_back(z[i]);

    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(1);
    Fsettings s; s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = nbins;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::SOFTNAN].rval = -7777.0;
    load_masked_test_roi_data(roidata, img.data(), msk.data(), img.size());
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f; f.calculate(roidata, s, ds);
    roidata.initialize_fvals(); f.save_value(roidata.fvals);
    auto g=[&](Nyxus::Feature2D fc){ return roidata.fvals[(int)fc][0]; };
    std::cout << "nbins="<<nbins
      << " BIN_SIZE="<<g(Nyxus::Feature2D::IH_BIN_SIZE)
      << " MIN_VAL="<<g(Nyxus::Feature2D::IH_MINIMUM_VAL)
      << " VAR_IDX="<<g(Nyxus::Feature2D::IH_VARIANCE_IDX)
      << " ENTROPY_IDX="<<g(Nyxus::Feature2D::IH_ENTROPY_IDX)
      << " UNIFORMITY_IDX="<<g(Nyxus::Feature2D::IH_UNIFORMITY_IDX)
      << " MEAN_IDX="<<g(Nyxus::Feature2D::IH_MEAN_IDX)
      << " ...\n";  // extend to all 20 + P25_IDX/P75_IDX
}
```
  Add `#include "scratch_ih_dump.h"` and `TEST(TEST_NYXUS, SCRATCH_IH_DUMP){ ih_dump_phantom(6); ih_dump_phantom(1); }` to `test_all.cc` temporarily.

- [ ] **Step 3: Build and run the dump.**
  Run: `cd /usr/axle/dev/nyxus && cmake --build build_gtest --target runAllTests -j && ./build_gtest/tests/runAllTests --gtest_filter='*SCRATCH_IH_DUMP*'`
  (Use the project's existing `build_gtest` setup — see prior local build in the repo notes.)
  Expected: prints Nyxus IH actuals for the phantom at the candidate bin counts.

- [ ] **Step 4: Reconcile config + index base.**
  Compare Nyxus actuals to the IBSI table from Step 1. Determine `IH_PHANTOM_NBINS` (the setting that reproduces the IBSI discretisation) and `IH_PHANTOM_INDEX_BASE`: check whether shift-invariant IDX features (VARIANCE_IDX, ENTROPY_IDX, UNIFORMITY_IDX, RANGE_IDX, IQR_IDX, MAD/rMAD/medAD_IDX, SKEWNESS_IDX, EXCESS_KURTOSIS_IDX) already match IBSI (offset-independent). If they match → config is correct. For MEAN_IDX, infer the index base (Nyxus 0-based vs IBSI 1-based → constant offset). Record findings in a comment.
  Expected: shift-invariant IDX features agree with IBSI within rel 1e-2. If they do **not**, stop and report — that is a config gap or a genuine Nyxus discrepancy (Global Constraints: no `src` fix here).

- [ ] **Step 5: Remove the scratch harness.**
  Delete `tests/scratch_ih_dump.h` and the temporary include + `SCRATCH_IH_DUMP` TEST from `test_all.cc`. Nothing from this task is committed except knowledge captured into Task 2's goldens/comment.

- [ ] **Step 6: Commit (docs only).**
  Record the reconciled config + index base as a note appended to the design doc's §6.

```bash
git add docs/superpowers/specs/2026-07-15-ibsi-ih-dispersion-vetting-design.md
git commit -m "docs(vetting): record reconciled IBSI IH phantom config + index base"
```

---

### Task 2: IDX oracle test vs IBSI consensus

**Files:**
- Create: `tests/test_intensity_histogram_ibsi.h`
- Modify: `tests/test_all.cc` (include + one TEST)

**Interfaces:**
- Consumes: `IH_PHANTOM_NBINS`, `ibsi_ih_phantom_golden`, `IH_PHANTOM_INDEX_BASE` (Task 1).
- Produces: `test_ih_dispersion_ibsi()` (extended in Task 3); the phantom-run helper `ih_ibsi_run(fvals, nbins)`.

- [ ] **Step 1: Write the failing test file.**
  Create `tests/test_intensity_histogram_ibsi.h`. `ibsi_ih_phantom_golden` values come from Task 1 (IBSI reference — fill the literals from the sourced table; the keys below are fixed):

```cpp
#pragma once
#include <gtest/gtest.h>
#include <unordered_map>
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// Provenance: IBSI (Zwanenburg et al. 2020) digital-phantom intensity-histogram
// (family 3.4) consensus values. Discretisation/config + index base recorded in
// docs/.../2026-07-15-ibsi-ih-dispersion-vetting-design.md §6. Values sourced in Task 1.
static const int IH_PHANTOM_NBINS = /*Task 1*/;
static std::unordered_map<std::string,double> ibsi_ih_phantom_golden = {
    // {"VARIANCE_IDX", <ibsi>}, {"SKEWNESS_IDX", <ibsi>}, ... (12 IBSI IH features)
};

static void ih_ibsi_run(std::vector<std::vector<double>>& fvals, int nbins) {
    std::vector<NyxusPixel> img, msk;
    for (auto* z : {ibsi_phantom_z1_intensity, ibsi_phantom_z2_intensity,
                    ibsi_phantom_z3_intensity, ibsi_phantom_z4_intensity})
        for (size_t i = 0; i < 20; i++) img.push_back(z[i]);
    for (auto* z : {ibsi_phantom_z1_mask, ibsi_phantom_z2_mask,
                    ibsi_phantom_z3_mask, ibsi_phantom_z4_mask})
        for (size_t i = 0; i < 20; i++) msk.push_back(z[i]);
    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(1);
    Fsettings s; s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = nbins;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::SOFTNAN].rval = -7777.0;
    load_masked_test_roi_data(roidata, img.data(), msk.data(), img.size());
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

static double ihg(const std::vector<std::vector<double>>& fv, Nyxus::Feature2D fc){ return fv[(int)fc][0]; }

// IDX dispersion/index features vs IBSI intensity-histogram consensus (12 with IBSI values).
void test_ih_dispersion_ibsi() {
    using F = Nyxus::Feature2D;
    std::vector<std::vector<double>> fv;
    ih_ibsi_run(fv, IH_PHANTOM_NBINS);
    auto chk = [&](const char* key, F fc){
        double gt = ibsi_ih_phantom_golden[key];
        if (std::abs(gt) < 1e-12) ASSERT_NEAR(ihg(fv,fc), gt, 1e-9) << key;
        else ASSERT_TRUE(agrees_gt(ihg(fv,fc), gt, 100.)) << key;  // rel 1e-2 (IBSI phantom tier)
    };
    chk("VARIANCE_IDX",                          F::IH_VARIANCE_IDX);
    chk("SKEWNESS_IDX",                          F::IH_SKEWNESS_IDX);
    chk("EXCESS_KURTOSIS_IDX",                   F::IH_EXCESS_KURTOSIS_IDX);
    chk("INTERQUANTILE_RANGE_IDX",               F::IH_INTERQUANTILE_RANGE_IDX);
    chk("RANGE_IDX",                             F::IH_RANGE_IDX);
    chk("MEAN_ABSOLUTE_DEVIATION_IDX",           F::IH_MEAN_ABSOLUTE_DEVIATION_IDX);
    chk("ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX",    F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX);
    chk("MEDIAN_ABSOLUTE_DEVIATION_IDX",         F::IH_MEDIAN_ABSOLUTE_DEVIATION_IDX);
    chk("COEFFICIENT_OF_VARIATION_IDX",          F::IH_COEFFICIENT_OF_VARIATION_IDX);
    chk("QUANTILE_COEFFICIENT_OF_DISPERSION_IDX",F::IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX);
    chk("ENTROPY_IDX",                           F::IH_ENTROPY_IDX);
    chk("UNIFORMITY_IDX",                        F::IH_UNIFORMITY_IDX);
    // ROBUST_MEAN_IDX has no IBSI feature -> covered analytically in Task 4.
}
```

- [ ] **Step 2: Register in `test_all.cc`.**
  Add near the other IH includes: `#include "test_intensity_histogram_ibsi.h"`. Add after `TEST_IH_DISPERSION_AND_INDEX_VALUES` is not present on main, so place after the IH block, before the HU block:

```cpp
TEST(TEST_NYXUS, TEST_IH_DISPERSION_IBSI)
{
	ASSERT_NO_THROW(test_ih_dispersion_ibsi());
}
```

- [ ] **Step 3: Build and run.**
  Run: `cmake --build build_gtest --target runAllTests -j && ./build_gtest/tests/runAllTests --gtest_filter='*TEST_IH_DISPERSION_IBSI*'`
  Expected: PASS (12 IDX features match IBSI consensus). If a shift-invariant feature fails → Task 1 config/offset was wrong; return to Task 1. If only CoV_IDX/QCoD_IDX fail → index-base offset unresolved (Step 4 of Task 1).

- [ ] **Step 4: Commit.**

```bash
git add tests/test_intensity_histogram_ibsi.h tests/test_all.cc
git commit -m "test(intensity-histogram): vet 12 IH _IDX features vs IBSI phantom consensus"
```

---

### Task 3: VAL transform anchoring

**Files:** Modify `tests/test_intensity_histogram_ibsi.h` (extend `test_ih_dispersion_ibsi`).

**Interfaces:** Consumes `ih_ibsi_run`, `ihg`, the phantom run's `IH_BIN_SIZE`/`IH_MINIMUM_VAL`. Produces the 6 VAL assertions anchored to the IBSI-vetted IDX quantities.

**Anchorability (finalised during self-review):** value = `a + b·index` with `b = binWidth`,
`a = IH_MINIMUM_VAL + 0.5·b`. Cleanly IBSI-anchorable from exposed features (**4 VAL**):
- Pure-scale spreads → `VAL = b·IDX`: `MEAN_ABSOLUTE_DEVIATION_VAL`,
  `ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL`, `MEDIAN_ABSOLUTE_DEVIATION_VAL`. (These use the per-bin
  deviation loop with unfloored `i`, so `deltaValue = b·deltaIndex` holds exactly.)
- Ratio via IBSI variance + exposed value-mean → `COEFFICIENT_OF_VARIATION_VAL = b·sqrt(VARIANCE_IDX)/MEAN_VAL`
  (`VARIANCE_IDX` is the IBSI anchor; `IH_MEAN_VAL` is exposed).

**NOT cleanly anchorable → analytic (Task 4):**
- `QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` — needs the sum `P75_VAL + P25_VAL`; Nyxus exposes only
  `P10_VAL`/`P90_VAL`.
- `INTERQUANTILE_RANGE_VAL` — `IQR_IDX = p75Index − p25Index` uses `getIndexOf()` (floored bin
  index), while `IQR_VAL = p75Value − p25Value` uses interpolated quantiles, so `IQR_VAL ≠ b·IQR_IDX`
  (verified empirically: 2.42604 vs 2.5 on the phantom). The `IQR_IDX`-floor / `IQR_VAL`-interpolate
  inconsistency is recorded as a Nyxus bug (see `docs/known-issues/`).

- [ ] **Step 1: Add the 4 IBSI-anchored VAL assertions** to the end of `test_ih_dispersion_ibsi()`:

```cpp
    // ---- VAL anchored to the IBSI-vetted IDX values (design §5) ----
    double b = ihg(fv, F::IH_BIN_SIZE);                 // binWidth
    // pure-scale spreads: VAL = b * IDX  (IQR_VAL excluded — quantile flooring breaks the identity)
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_MEAN_ABSOLUTE_DEVIATION_VAL),
                          b*ihg(fv,F::IH_MEAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL),
                          b*ihg(fv,F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_MEDIAN_ABSOLUTE_DEVIATION_VAL),
                          b*ihg(fv,F::IH_MEDIAN_ABSOLUTE_DEVIATION_IDX), 1e4));
    // CoV_VAL = std_VAL / mean_VAL = b*sqrt(VARIANCE_IDX) / MEAN_VAL  (VARIANCE_IDX = IBSI anchor)
    double cov_val_expected = b*std::sqrt(ihg(fv,F::IH_VARIANCE_IDX)) / ihg(fv,F::IH_MEAN_VAL);
    ASSERT_TRUE(agrees_gt(ihg(fv,F::IH_COEFFICIENT_OF_VARIATION_VAL), cov_val_expected, 1e4));
```

- [ ] **Step 2: Build and run.**
  Run: `cmake --build build_gtest --target runAllTests -j && ./build_gtest/tests/runAllTests --gtest_filter='*TEST_IH_DISPERSION_IBSI*'`
  Expected: PASS — the 4 IBSI-anchored VAL features equal their transforms.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_intensity_histogram_ibsi.h
git commit -m "test(intensity-histogram): anchor 4 IH _VAL features to IBSI-vetted IDX"
```

---

### Task 4: Analytic checks — robust-window fixture + QCoD_VAL

**Files:** Modify `tests/test_intensity_histogram_ibsi.h` (add fixture + `test_ih_dispersion_robust_analytic`); modify `test_all.cc` (register).

**Interfaces:** Produces `intensityHistogramRobustData` fixture + `test_ih_dispersion_robust_analytic()` covering the 4 `oracle=analytic` features — `ROBUST_MEAN_IDX`, `ROBUST_MEAN_VAL` (no IBSI feature), `QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` and `INTERQUANTILE_RANGE_VAL` (not IBSI-anchorable — quantile flooring) — plus the robust-MAD trimming path. All goldens are hand-computed (carried from PR 367, derived independently of `intensity_histogram.cpp`).

- [ ] **Step 1: Add the discriminating fixture + test.** 17-px ROI, N=5, `freq={1,5,6,4,1}`, robust window `[p10Idx,p90Idx]` trims tail bins (robust ≠ full). Goldens are hand-derived (carried from PR 367, which computed them independently of `intensity_histogram.cpp`):

```cpp
static const NyxusPixel intensityHistogramRobustData[] = {
    {0,0,0},
    {1,0,10},{2,0,10},{3,0,10},{4,0,10},{5,0,10},
    {6,0,20},{7,0,20},{8,0,20},{9,0,20},{10,0,20},{11,0,20},
    {12,0,30},{13,0,30},{14,0,30},{15,0,30},
    {16,0,40}
};

void test_ih_dispersion_robust_analytic() {
    using F = Nyxus::Feature2D;
    Fsettings s; s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = 5;
    s[(int)NyxSetting::IBSI].bval = true;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::SOFTNAN].rval = -7777.0;
    Dataset ds; ds.dataset_props.push_back(SlideProps("",""));
    LR roidata(100); roidata.slide_idx = -1;
    load_test_roi_data(roidata, intensityHistogramRobustData,
                       sizeof(intensityHistogramRobustData)/sizeof(NyxusPixel));
    roidata.make_nonanisotropic_aabb();
    IntensityHistogramFeatures f; ASSERT_NO_THROW(f.calculate(roidata, s, ds));
    roidata.initialize_fvals(); f.save_value(roidata.fvals);
    auto& fv = roidata.fvals;
    // Hand-computed goldens (robust window strictly trims tails -> robust != full):
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL][0], 4.977777778, 1e4));
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_VAL][0], 19.46666667, 1e4));         // oracle=analytic
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX][0], 0.6222222222, 1e4));
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_ROBUST_MEAN_IDX][0], 1.933333333, 1e4));         // oracle=analytic
    // QCoD_VAL: not IBSI-anchorable (needs unexposed P25/P75 sum) -> analytic golden here.
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL][0], 0.3178294574, 1e4)); // oracle=analytic
    // IQR_VAL: not IBSI-anchorable (IQR_IDX bin-floored vs IQR_VAL interpolated) -> analytic golden here.
    ASSERT_TRUE(agrees_gt(fv[(int)F::IH_INTERQUANTILE_RANGE_VAL][0], 12.3, 1e4)); // oracle=analytic
}
```

- [ ] **Step 2: Register in `test_all.cc`.**

```cpp
TEST(TEST_NYXUS, TEST_IH_DISPERSION_ROBUST_ANALYTIC)
{
	ASSERT_NO_THROW(test_ih_dispersion_robust_analytic());
}
```

- [ ] **Step 3: Build and run.**
  Run: `cmake --build build_gtest --target runAllTests -j && ./build_gtest/tests/runAllTests --gtest_filter='*TEST_IH_DISPERSION_ROBUST_ANALYTIC*'`
  Expected: PASS. Confirms `robust != full` (robust-mean 19.47 vs full-mean would differ) and exercises the trimming path.

- [ ] **Step 4: Commit.**

```bash
git add tests/test_intensity_histogram_ibsi.h tests/test_all.cc
git commit -m "test(intensity-histogram): analytic robust-window fixture for IH robust-mean/rMAD"
```

---

### Task 5: Registry, config recipe, coverage report

**Files:** Modify `tests/vetting/config_recipes.md`, `tests/vetting/oracle_coverage.csv`; regenerate `tests/vetting/coverage_report.md`.

**Interfaces:** Consumes nothing from tests at runtime; encodes the vetting outcome in the registry.

- [ ] **Step 1: Add the `ih.ibsi_fbn` recipe** after `ih.mirp_fbn` in `config_recipes.md`:

```markdown
## ih.ibsi_fbn
- Fixed-bin discretised intensity histogram (IBSI IH family 3.4) on the IBSI digital phantom,
  GREYDEPTH = <IH_PHANTOM_NBINS>, IBSI=true. Oracle: `ibsi`. Used by: `test_intensity_histogram_ibsi.h`.
- `IH_*_IDX` vet directly against IBSI IH consensus (index/grey-level domain). `IH_*_VAL` are the
  same statistics over bin centers = affine transform of the IDX distribution (VAL = binWidth·IDX
  for spreads; +minVal offset for locations), so they are anchored to the IBSI-vetted IDX values.
  `ROBUST_MEAN_*` have no IBSI feature -> analytic (see test_ih_dispersion_robust_analytic).
```

- [ ] **Step 2: Flip the 20 registry rows** with a CSV-safe script (run once):

```python
import csv
PATH="tests/vetting/oracle_coverage.csv"
IBSI = {  # 16 rows -> oracle=ibsi (12 IDX with IBSI consensus + 4 IBSI-anchored VAL)
  "IH_VARIANCE_IDX","IH_SKEWNESS_IDX","IH_EXCESS_KURTOSIS_IDX","IH_INTERQUANTILE_RANGE_IDX",
  "IH_RANGE_IDX","IH_MEAN_ABSOLUTE_DEVIATION_IDX","IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX",
  "IH_MEDIAN_ABSOLUTE_DEVIATION_IDX","IH_COEFFICIENT_OF_VARIATION_IDX",
  "IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX","IH_ENTROPY_IDX","IH_UNIFORMITY_IDX",
  "IH_MEAN_ABSOLUTE_DEVIATION_VAL",
  "IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL","IH_MEDIAN_ABSOLUTE_DEVIATION_VAL",
  "IH_COEFFICIENT_OF_VARIATION_VAL",
}
# 4 rows -> oracle=analytic: robust-mean pair (no IBSI feature) + QCoD_VAL & IQR_VAL (quantile-floored,
# not IBSI-anchorable). IQR_IDX-floor/IQR_VAL-interpolate inconsistency filed as a Nyxus bug (docs/known-issues/).
ANALYTIC = {"IH_ROBUST_MEAN_IDX","IH_ROBUST_MEAN_VAL",
            "IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL","IH_INTERQUANTILE_RANGE_VAL"}
NOTE_IBSI="vetted by test_ih_dispersion_ibsi (IBSI IH phantom consensus for _IDX; the 4 anchorable _VAL via affine VAL=transform(IDX), recipe ih.ibsi_fbn)"
NOTE_AN="vetted by test_ih_dispersion_robust_analytic (hand-computed on the robust-window discriminating fixture; no clean IBSI anchor - robust-mean has no IBSI feature, QCoD_VAL/IQR_VAL are quantile-floored, see docs/known-issues/)"
rows=list(csv.DictReader(open(PATH))); cols=rows[0].keys()
n=0
for r in rows:
    if r["family"]!="intensity_histogram" or r["status"].strip()!="untested": continue
    f=r["feature"]
    if f in IBSI: r.update(status="vetted",oracle="ibsi",config_recipe="ih.ibsi_fbn",
                           current_test="test_intensity_histogram_ibsi.h",notes=NOTE_IBSI); n+=1
    elif f in ANALYTIC: r.update(status="vetted",oracle="analytic",config_recipe="ih.ibsi_fbn",
                           current_test="test_intensity_histogram_ibsi.h",notes=NOTE_AN); n+=1
assert n==20, n
csv.DictWriter(open(PATH,"w",newline=""),fieldnames=list(cols),quoting=csv.QUOTE_MINIMAL).writerows(
    [dict(zip(cols,cols))]+rows) if False else None
w=csv.DictWriter(open(PATH,"w",newline=""),fieldnames=list(cols),quoting=csv.QUOTE_MINIMAL)
w.writeheader(); w.writerows(rows); print("updated",n)
```
  Run: `cd /usr/axle/dev/nyxus && python3 - <<'PY'` … `PY` (paste the block). Expected: `updated 20`.

- [ ] **Step 3: Regenerate the report + validate.**
  Run: `python3 tests/vetting/check_coverage.py --write && python3 tests/vetting/check_coverage.py --check; echo $?`
  Expected: report written; `--check` exit `0`; `intensity_histogram` untested drops to `1` (the `HISTOGRAM` placeholder), IH vetted `46 -> 66`? (verify actual delta: +20 features to vetted).

- [ ] **Step 4: Commit.**

```bash
git add tests/vetting/config_recipes.md tests/vetting/oracle_coverage.csv tests/vetting/coverage_report.md
git commit -m "vetting(registry): promote 20 IH dispersion/index features (18 ibsi + 2 analytic)"
```

---

### Task 6: Full-suite verification

- [ ] **Step 1: Build + run the whole gtest suite.**
  Run: `cmake --build build_gtest --target runAllTests -j && ./build_gtest/tests/runAllTests`
  Expected: all tests PASS including the two new IH cases; no regression in the prior count.

- [ ] **Step 2: Final registry validation.**
  Run: `python3 tests/vetting/check_coverage.py --check; echo $?`
  Expected: `0`.

- [ ] **Step 3: Confirm no `src/` changes.**
  Run: `git diff --name-only polus_origin/main..HEAD | grep '^src/' && echo "SRC CHANGED (bad)" || echo "test-only (good)"`
  Expected: `test-only (good)`.
