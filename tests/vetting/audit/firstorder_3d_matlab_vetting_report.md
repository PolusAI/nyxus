# 3D first-order vs the MATLAB oracle — vetting report

`tests/test_3d_firstorder_matlab.h` carries 35 goldens for the 3D first-order family. Every one is
GNU Octave's own value at full precision, and every one is asserted against live Nyxus output by a
case in `test_all.cc`. This report records what those assertions claim and how to re-derive them.
What the audit found in the table they replaced is in the [appendix](#appendix-what-the-audit-found).

## What the file asserts

Two bands, each set from the measured Nyxus-vs-oracle residual rather than from a round number.
`agrees_gt`'s third argument is a **divisor** (`tolerance = golden / frac`), so a larger number is a
tighter band.

| band | value | features | measured worst |
|---|---|---|---|
| `FO3D_MATLAB_EXACT` | rel 1e-9 | 25 — the statistics where Nyxus and Octave compute the same quantity | 6.6e-14 (`3EXCESS_KURTOSIS`) |
| `FO3D_MATLAB_BINNED` | rel 5e-3 | 10 — `3P01/10/25/75/90/99`, `3INTERQUARTILE_RANGE`, `3QCOD`, `3ROBUST_MEAN`, `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | 2.3e-3 (`3P01`) |

Eighteen of the 35 also carry a live Nyxus baseline in `test_3d_firstorder_coverage.h`, so their
residual can be read straight off the two pinned tables with no build:

| feature | Nyxus | Octave | rel | band |
|---|---:|---:|---:|---|
| `3P01` | 1039.3829596413 | 1037 | 2.3e-3 | BINNED |
| `3QCOD` | 0.2572485182717 | 0.2574244913434 | 6.8e-4 | BINNED |
| `3P25` | 1469.7943925234 | 1469 | 5.4e-4 | BINNED |
| `3ROBUST_MEAN` | 1977.5189642597 | 1976.5703930354 | 4.8e-4 | BINNED |
| `3P75` | 2487.9072847682 | 2487.5 | 1.6e-4 | BINNED |
| `3P99` | 3002.3047021944 | 3002 | 1.0e-4 | BINNED |
| `3EXCESS_KURTOSIS` | −1.2127631603215 | −1.2127631603216 | 6.6e-14 | EXACT |
| `3COV` | 0.2948620704346 | 0.2948620704346 | 0 | EXACT |
| `3HYPERFLATNESS` | 3.8027657005973 | 3.8027657005973 | 0 | EXACT |
| `3HYPERSKEWNESS` | 0.3200133261552 | 0.3200133261552 | 0 | EXACT |
| `3INTEGRATED_INTENSITY` | 544286216 | 544286216 | 0 | EXACT |
| `3MEDIAN_ABSOLUTE_DEVIATION` | 507.1238048041 | 507.1238048041 | 0 | EXACT |
| `3MODE` | 1279 | 1279 | 0 | EXACT |
| `3STANDARD_DEVIATION` | 584.8055640696 | 584.8055640696 | 0 | EXACT |
| `3STANDARD_DEVIATION_BIASED` | 584.8044985851 | 584.8044985851 | 0 | EXACT |
| `3STANDARD_ERROR` | 1.1163339190447 | 1.1163339190447 | 0 | EXACT |
| `3UNIFORMITY_PIU` | 50.592885375494 | 50.592885375494 | 0 | EXACT |
| `3VARIANCE_BIASED` | 341996.30156538 | 341996.30156538 | 0 | EXACT |

For the other 17 the only pinned Nyxus value in the tree is the golden itself, so their evidence is
the assertion passing at the stated band rather than a number quoted here.

`3SKEWNESS` and `3KURTOSIS` are a direct comparison rather than a convention match: `Moments4`
computes `sqrt(n)·M3/M2^1.5` and `n·M4/M2²`, which are exactly Octave's population `skewness()` and
`kurtosis()`.

## Reproduction

| | |
|---|---|
| generator | `tests/vetting/oracles/gen_firstorder3d_matlab.py` + `.m` |
| oracle | GNU Octave 11.3.0 + `statistics` package (SPEC §4: the `matlab` token names the semantics) |
| reader | SimpleITK 2.3.1 under Python 3.8.20, NumPy 1.23.5 |
| fixture | `tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57, n = 274432 voxels |
| Nyxus settings | default-constructed `Fsettings` → `DEFAULT_NUM_HISTO_BINS` = 24 |

Re-verification is two halves, and both are needed. The generator closes the first — that each pin
in the header equals the oracle — and exits non-zero otherwise:

```
python gen_firstorder3d_matlab.py --octave <octave-cli>
```

The C++ suite closes the second, that Nyxus reproduces those pins inside the declared band:

```
runAllTests --gtest_filter=*3D_FIRSTORDER*
```

Last run: `35 pins, 0 mismatched, 0 unproducible, 0 unpinned oracle values` from the generator, and
`53 tests ... PASSED` from the suite.

### The loader domain, and why the oracle runs in it

Nyxus does not featurize the stored voxels. `NiftiLoader::unhounsfield` (`src/nyx/raw_nifti.h`)
scans the **whole volume** and, when its minimum is negative, shifts every voxel by `−min` before
the cast to the unsigned buffer truncates it. `ut_inten.nii` stores `[−1024, 2000]`, so the label-57
ROI Nyxus measures is `trunc(stored + 1024)`, spanning `[1024, 3024]`.

The oracle reproduces that transform and runs on its output. Running on the stored voxels instead
puts every location statistic exactly 1024 off and leaves every shift-invariant one untouched —
which is a loader question, not a first-order one, and is the same signature as the reported CT
defect. Confirmation that the reproduction is exact: `sum = 544286216`, matching the live coverage
baseline in `test_3d_firstorder_coverage.h` to the digit.

**What this does and does not vet.** It vets the *statistics* against MATLAB semantics. It does not
vet the loader transform, which no assertion in this family covers.

### Range and identity checks

Run over the whole pin set by the generator, which exits non-zero on any failure. All twelve pass,
as does the reverse check — an oracle value the header pins nothing for — which is clean at 0.

| check | result |
|---|---|
| `3UNIFORMITY` in [0,1] | pass |
| `3ENTROPY ≥ 0` | pass |
| `3UNIFORMITY_PIU` in [0,100] | pass |
| `3RANGE == 3MAX − 3MIN` | pass |
| `3STANDARD_DEVIATION == sqrt(3VARIANCE)` | pass |
| `3STANDARD_DEVIATION_BIASED == sqrt(3VARIANCE_BIASED)` | pass |
| `3EXCESS_KURTOSIS == 3KURTOSIS − 3` | pass |
| `3MEAN == 3INTEGRATED_INTENSITY / n` | pass |
| `3INTERQUARTILE_RANGE == 3P75 − 3P25` | pass |
| `3QCOD == (3P75−3P25)/(3P75+3P25)` | pass |
| `3P10 ≤ 3ROBUST_MEAN ≤ 3P90` | pass |
| `3MIN ≤ 3P01 ≤ 3P99 ≤ 3MAX` | pass |

## Why the percentile features are oracle tests, not regression pins

`3P01`…`3P99`, `3INTERQUARTILE_RANGE` and `3QCOD` come from the 100-bin interpolated histogram in
`TrivialHistogram::calc_percentiles`, not from an order statistic. The oracle deliberately runs
MATLAB `prctile` — the definition Nyxus approximates — instead of re-implementing the estimator, so
the comparison measures the approximation rather than confirming it against itself. (For contrast,
the pre-existing harness under `octave/oracle_3d/` re-implements the binned estimator in Octave and
reports 15-digit agreement — that number measures nothing.)

That is why their band is 5e-3 rather than 1e-9, and it is not a reason to demote them. What
separates an oracle test from a regression test is where the pin comes from, not how tight the band
is. A regression pin is Nyxus' own output, so it can only detect change — it would lock in whatever
the estimator does, right or wrong. These ten pins are MATLAB `prctile` values computed with no
knowledge of Nyxus, so asserting the feature within 5e-3 of one is a correctness claim: the binned
estimator tracks the true order statistic to better than a quarter of a percent. Demoting them would
trade that claim for "Nyxus still equals Nyxus".

## What the oracle does not back

| feature | reason |
|---|---|
| `3COVERED_IMAGE_INTENSITY_RANGE` | needs `SlideProps` whole-slide min/max, so the generator cannot produce it from the ROI voxel vector at all. Correctly `status=regression`, asserted in `test_3d_firstorder_regression.h` against its own pin (the live baseline 1.0002043207290587) rather than borrowing this file's map |

`3ENTROPY` and `3UNIFORMITY` come from the custom-resolution bin histogram, whose bin count is a
Nyxus setting. That makes them recipe-dependent, not unbackable. The oracle is handed the bin count
the way a PyRadiomics `binCount` would be and then implements the estimator itself in Octave —
equal-width binning of `[min,max]`, `-Sum p*log2(p)` and `Sum p^2` — so the comparison is still
against an independently computed value. Both pins are Octave's, and Nyxus reproduces them inside
the `_EXACT` band.

## Negative control

The bands are not decorative. Both of the wrong values the audit turned up were planted back into
the regenerated header and the suite re-run:

| planted | Nyxus actual | difference | tolerance at rel 1e-9 / 5e-3 | tolerance at ±10% |
|---|---:|---:|---|---|
| `3ENTROPY` = 4.24 | 4.57954 | 0.339542 | 4.24e-09 → **fails** | 0.424 → would pass |
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION` = 392.98 | 406.982 | 14.0017 | 1.9649 → **fails** | 39.298 → would pass |

Both failed and named the feature; the values were then restored and all 36 cases pass. The run also
confirms Nyxus' own output matches the oracle — 4.57954 against 4.5795418896949416, and 406.982
against the 406.9817011068 the binned window produces.

## Provenance

SPEC §6.4 makes provenance mandatory. The generator, the oracle version, the fixture, the recipe and
this report are recorded in the header's PROVENANCE block and in the Reproduction table above. SPEC
§4 names this table as the one legacy 3D table predating the rule that `matlab` means Octave.

---

# Appendix: what the audit found

Everything below describes the file **as found**, before the goldens above replaced it. It is kept
for the record of what a table can hide; none of it describes the current state.

The table had never been executed: `test_all.cc` did not include the header, and its only includer
was the equally unreachable `tests/test_3d_firstorder_regression.h`. Eighteen registry rows read
`status=vetted, oracle=matlab` on the strength of it. Every golden was compared against a fresh
Octave run on the fixture the file's own accessor names.

## 29 of 36 pins reproduced

Their residual was bounded by the pins' own printed precision — most carried three significant
figures, so `rel` sat at 1e-3 to 1e-7 rather than at the agreement's true floor.

| feature | header pin | Octave | rel | verdict |
|---|---:|---:|---:|---|
| `3MIN` | 1024 | 1024 | 0 | agrees |
| `3MAX` | 3024 | 3024 | 0 | agrees |
| `3RANGE` | 2000 | 2000 | 0 | agrees |
| `3MEDIAN` | 1964.5 | 1964.5 | 0 | agrees |
| `3MODE` | 1279 | 1279 | 0 | agrees |
| `3INTEGRATED_INTENSITY` | 544286000 | 544286216 | 4.0e-7 | agrees |
| `3ROOT_MEAN_SQUARED` | 2067.74 | 2067.740504 | 2.4e-7 | agrees |
| `3MEAN` | 1983.32 | 1983.319059 | 4.7e-7 | agrees |
| `3VARIANCE_BIASED` | 341996 | 341996.3016 | 8.8e-7 | agrees |
| `3MEAN_ABSOLUTE_DEVIATION` | 507.29 | 507.2894758 | 1.0e-6 | agrees |
| `3VARIANCE` | 341998 | 341997.5478 | 1.3e-6 | agrees |
| `3ENERGY` | 1.17335e12 | 1.173347955e12 | 1.7e-6 | agrees |
| `3MEDIAN_ABSOLUTE_DEVIATION` | 507.12 | 507.1238048 | 7.5e-6 | agrees |
| `3STANDARD_DEVIATION` | 584.81 | 584.8055641 | 7.6e-6 | agrees |
| `3STANDARD_DEVIATION_BIASED` | 584.8 | 584.8044986 | 7.7e-6 | agrees |
| `3UNIFORMITY_PIU` | 50.59 | 50.59288538 | 5.7e-5 | agrees |
| `3HYPERSKEWNESS` | 0.32 | 0.3200133262 | 4.2e-5 | agrees |
| `3P99` | 3002.3 | 3002 | 1.0e-4 | agrees |
| `3P75` | 2487.91 | 2487.5 | 1.6e-4 | agrees |
| `3P90` | 2808.61 | 2808 | 2.2e-4 | agrees |
| `3INTERQUARTILE_RANGE` | 1018.11 | 1018.5 | 3.8e-4 | agrees |
| `3P25` | 1469.79 | 1469 | 5.4e-4 | agrees |
| `3HYPERFLATNESS` | 3.8 | 3.802765701 | 7.3e-4 | agrees |
| `3P10` | 1189.05 | 1188 | 8.8e-4 | agrees |
| `3P01` | 1039.38 | 1037 | 2.3e-3 | agrees |
| `3EXCESS_KURTOSIS` | −1.21 | −1.21276316 | 2.3e-3 | agrees |
| `3STANDARD_ERROR` | 1.12 | 1.116333919 | 3.3e-3 | agrees |
| `3SKEWNESS` | 0.075 | 0.07469052913 | 4.1e-3 | agrees |
| `3KURTOSIS` | 1.78 | 1.78723684 | 4.0e-3 | agrees |

## Four pins did not hold up

| feature | header pin | Nyxus | true statistic | rel vs Nyxus | at ±10% |
|---|---:|---:|---:|---:|---|
| `3ROBUST_MEAN` | **0** | 1977.5189642597 | 1976.5703930354 | 1.0 | **FAILS** |
| `3UNIFORMITY` | **307211000** | 0.041991745610 | — | 7.3e9 | **FAILS** |
| `3ENTROPY` | **4.24** | 4.579541890 | — | 7.4e-2 | passes |
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | **392.98** | 406.9817011068 | 407.4270631429 | 3.4e-2 | passes |

1. **`3ROBUST_MEAN = 0` was the pre-fix bug value.** `3d_intensity.cpp` computes the mean over the
   `[P10,P90]` window and the live coverage baseline pins `1977.5189642596645`, which the binned
   window reproduces exactly. The pin also failed the identity `P10 ≤ ROBUST_MEAN ≤ P90`.

2. **`3UNIFORMITY = 307211000` violated the range its definition forces.** `get_stats()` returns
   `Σp²`, which lies in `[0,1]`. The pin was `Σcount²` — unnormalized — and reproduces exactly as
   `sum(count²)` over a **256-bin** histogram (3.07211e8). It was stale in both normalization and
   bin count.

3. **`3ENTROPY = 4.24` did not reproduce at the default 24 bins** (4.579542). Swept across bin
   counts it lands on **19 bins** (4.243139) — which is not a Nyxus default and does not match the
   256 bins `3UNIFORMITY` implies, so the two pins came from different configurations.

4. **`3ROBUST_MEAN_ABSOLUTE_DEVIATION = 392.98` matched neither estimator** — 3.4% off the binned
   window (406.98) and 3.5% off the exact one (407.43). `histogram.h` records a "p10/p90 robust-MAD
   fix"; the pin predated it.

### Two of the four survived the band the file asserted at

Both oracle files asserted with `agrees_gt(..., 10.)`, a **±10% band** on all 35 assertions. Against
a measured agreement of 1e-6 to 1e-7 on the exact statistics that was five to six orders of
magnitude too loose, and it is what let a 7.4%-wrong `3ENTROPY` and a 3.4%-wrong
`3ROBUST_MEAN_ABSOLUTE_DEVIATION` pass as "agreed".

## Two pins were too coarse to assert anything

`3COV` was pinned at **one** significant figure (0.3 against 0.2948620704, rel 1.7e-2) and `3QCOD`
at two (0.26 against 0.2574244913, rel 9.9e-3). Both failed an identity check purely on rounding:
`3QCOD == (3P75−3P25)/(3P75+3P25)` missed by 1.1e-2 using the file's own percentile pins. A golden
printed to fewer digits than the tolerance it is asserted at cannot carry a vetting claim.

## Range and identity checks, as found

| check | result |
|---|---|
| `3UNIFORMITY` in [0,1] | **FAIL** — 307211000 |
| `3QCOD == (3P75−3P25)/(3P75+3P25)` | **FAIL** — rounding, 1.1e-2 |
| `3P10 ≤ 3ROBUST_MEAN ≤ 3P90` | **FAIL** — pin is 0 |
| the other nine | pass |

## Provenance, which the file never recorded

SPEC §6.4 makes provenance mandatory and the header said outright that it had none: no MATLAB
version, no config, no generator. `not_covered.md` §C tracks it.

## What the audit changed

The table was regenerated from the Octave run at full precision, the file was wired into
`test_all.cc` so its 35 assertions execute, and the single ±10% band was replaced by the two bands
at the top of this report. `3COVERED_IMAGE_INTENSITY_RANGE` moved to
`test_3d_firstorder_regression.h` with its own pin, and the workflow mock both files share moved to
`test_3d_firstorder_common.h`, so the drift guard no longer includes the oracle file to reach it.
