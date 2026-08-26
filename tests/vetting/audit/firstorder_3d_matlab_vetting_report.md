# 3D first-order vs the MATLAB oracle — vetting report

Every golden in `tests/test_3d_firstorder_matlab.h` compared against a fresh GNU Octave run on the
fixture the file's own accessor names. The table had never been executed: `test_all.cc` does not
include the header, and its only includer is the equally unreachable
`tests/test_3d_firstorder_regression.h`. Eighteen registry rows read `status=vetted, oracle=matlab`
on the strength of it.

## Reproduction

| | |
|---|---|
| generator | `tests/vetting/oracles/gen_firstorder3d_matlab.py` + `.m` |
| oracle | GNU Octave 11.3.0 + `statistics` package (SPEC §4: the `matlab` token names the semantics) |
| reader | SimpleITK 2.3.1 under Python 3.8.20, NumPy 1.23.5 |
| fixture | `tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57, n = 274432 voxels |
| Nyxus settings | default-constructed `Fsettings` → `DEFAULT_NUM_HISTO_BINS` = 24 |

```
python gen_firstorder3d_matlab.py --octave <octave-cli>
```

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

## Result

29 of 36 pins reproduce. Their residual is bounded by the pins' own printed precision — most carry
three significant figures, so `rel` sits at 1e-3 to 1e-7 rather than at the agreement's true floor.

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

`3SKEWNESS` and `3KURTOSIS` are a direct comparison rather than a convention match: `Moments4`
computes `sqrt(n)·M3/M2^1.5` and `n·M4/M2²`, which are exactly Octave's population `skewness()` and
`kurtosis()`.

### The percentile estimator is not circular here

`3P01`…`3P99`, `3INTERQUARTILE_RANGE` and `3QCOD` come from the 100-bin interpolated histogram in
`TrivialHistogram::calc_percentiles`, not from an order statistic. The oracle deliberately runs
MATLAB `prctile` — the definition Nyxus approximates — instead of re-implementing the estimator, so
the comparison measures the approximation rather than confirming it against itself.

Measured, the approximation is good: the worst divergence is `3P01` at **2.3e-3** and the rest are
1e-4 to 9e-4. (For contrast, the pre-existing harness under `octave/oracle_3d/` re-implements the
binned estimator in Octave and reports 15-digit agreement — that number measures nothing.)

**Why they stay oracle tests rather than being demoted to regression.** What separates the two is
where the pin comes from, not how tight the band is. A regression pin is Nyxus' own output, so it
can only detect change — it would lock in whatever the estimator does, right or wrong. These ten
pins are MATLAB `prctile` values computed with no knowledge of Nyxus, so asserting the feature
within 5e-3 of one is a correctness claim: the binned estimator tracks the true order statistic to
better than a quarter of a percent. Demoting them would trade that claim for "Nyxus still equals
Nyxus". The band is set from the measured residual rather than from a round number, and the negative
control below shows it still catches the two defects the old ±10% let through.

## Four pins do not hold up

| feature | header pin | Nyxus today | true statistic | rel vs Nyxus | at ±10% |
|---|---:|---:|---:|---:|---|
| `3ROBUST_MEAN` | **0** | 1977.5189642597 | 1976.5703930354 | 1.0 | **FAILS** |
| `3UNIFORMITY` | **307211000** | 0.041991745610 | — | 7.3e9 | **FAILS** |
| `3ENTROPY` | **4.24** | 4.579541890 | — | 7.4e-2 | passes |
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION` | **392.98** | 406.9817011068 | 407.4270631429 | 3.4e-2 | passes |

1. **`3ROBUST_MEAN = 0` is the pre-fix bug value.** `3d_intensity.cpp` now computes the mean over
   the `[P10,P90]` window and the live coverage baseline pins `1977.5189642596645`, which the binned
   window reproduces exactly. The pin also fails the identity `P10 ≤ ROBUST_MEAN ≤ P90`.

2. **`3UNIFORMITY = 307211000` violates the range its definition forces.** `get_stats()` returns
   `Σp²`, which lies in `[0,1]`. The pin is `Σcount²` — unnormalized — and reproduces exactly as
   `sum(count²)` over a **256-bin** histogram (3.07211e8). It is stale in both normalization and bin
   count.

3. **`3ENTROPY = 4.24` does not reproduce at the default 24 bins** (4.579542). Swept across bin
   counts it lands on **19 bins** (4.243139) — which is not a Nyxus default and does not match the
   256 bins `3UNIFORMITY` implies, so the two pins come from different configurations.

4. **`3ROBUST_MEAN_ABSOLUTE_DEVIATION = 392.98` matches neither estimator** — 3.4% off the binned
   window (406.98) and 3.5% off the exact one (407.43). `histogram.h` records a "p10/p90 robust-MAD
   fix"; the pin predates it.

### Two of the four survive the band the file asserts at

Both oracle files assert with `agrees_gt(..., 10.)`. That third argument is a **divisor** —
`tolerance = golden / 10` — so it is a **±10% band** on all 35 assertions. Against a measured
agreement of 1e-6 to 1e-7 on the exact statistics, that is five to six orders of magnitude too
loose, and it is what lets a 7.4%-wrong `3ENTROPY` and a 3.4%-wrong `3ROBUST_MEAN_ABSOLUTE_DEVIATION`
pass as "agreed".

## One pin the oracle cannot back

| feature | reason |
|---|---|
| `3COVERED_IMAGE_INTENSITY_RANGE` | needs `SlideProps` whole-slide min/max, so the generator cannot produce it from the ROI voxel vector at all. Correctly `status=regression`, and its pin (1.0) is a rounding of the live baseline 1.0002043207290587 |

`3ENTROPY` and `3UNIFORMITY` come from the custom-resolution bin histogram, whose bin count is a
Nyxus setting. That makes them recipe-dependent, not unbackable. The oracle is handed the bin count
the way a PyRadiomics `binCount` would be and then implements the estimator itself in Octave —
equal-width binning of `[min,max]`, `-Sum p*log2(p)` and `Sum p^2` — so the comparison is still
against an independently computed value. Both regenerated pins are Octave's, and Nyxus reproduces
them inside the `_EXACT` band. What made the old pins wrong was that their recipe was never
recorded: 256 unnormalized bins for `3UNIFORMITY`, 19 bins for `3ENTROPY`.

## Two pins are too coarse to assert anything

`3COV` is pinned at **one** significant figure (0.3 against 0.2948620704, rel 1.7e-2) and `3QCOD` at
two (0.26 against 0.2574244913, rel 9.9e-3). Both fail an identity check purely on rounding:
`3QCOD == (3P75−3P25)/(3P75+3P25)` misses by 1.1e-2 using the file's own percentile pins. A golden
printed to fewer digits than the tolerance it is asserted at cannot carry a vetting claim.

## Mechanical range and identity checks

Run over the whole pin set by the generator, which exits non-zero on any failure.

| check | result |
|---|---|
| `3UNIFORMITY` in [0,1] | **FAIL** — 307211000 |
| `3QCOD == (3P75−3P25)/(3P75+3P25)` | **FAIL** — rounding, 1.1e-2 |
| `3P10 ≤ 3ROBUST_MEAN ≤ 3P90` | **FAIL** — pin is 0 |
| `3ENTROPY ≥ 0` | pass |
| `3UNIFORMITY_PIU` in [0,100] | pass |
| `3RANGE == 3MAX − 3MIN` | pass |
| `3STANDARD_DEVIATION == sqrt(3VARIANCE)` | pass |
| `3STANDARD_DEVIATION_BIASED == sqrt(3VARIANCE_BIASED)` | pass |
| `3EXCESS_KURTOSIS == 3KURTOSIS − 3` | pass |
| `3MEAN == 3INTEGRATED_INTENSITY / n` | pass |
| `3INTERQUARTILE_RANGE == 3P75 − 3P25` | pass |
| `3MIN ≤ 3P01 ≤ 3P99 ≤ 3MAX` | pass |

The generator also runs the reverse check — an oracle value the header pins nothing for — which is
clean at 0.

## What changed

The table is regenerated from the Octave run at full precision, and the file is wired into
`test_all.cc`, so its 35 assertions execute for the first time. Two bands replace the single ±10%:

| band | value | features | measured worst |
|---|---|---|---|
| `FO3D_MATLAB_EXACT` | rel 1e-9 | 25 | 6.6e-14 |
| `FO3D_MATLAB_BINNED` | rel 5e-3 | the 10 percentile-derived | 2.3e-3 (`3P01`) |

`3COVERED_IMAGE_INTENSITY_RANGE` moved to `test_3d_firstorder_regression.h` with its own pin rather
than borrowing this file's map, and the workflow mock both files share now lives in
`test_3d_firstorder_common.h`, so the drift guard no longer includes the oracle file to reach it.

### Negative control

The band change is what catches the two defects that would otherwise have passed. Both wrong values
were planted back into the regenerated header and the suite re-run:

| planted | Nyxus actual | difference | tolerance at rel 1e-9 / 5e-3 | tolerance at ±10% |
|---|---:|---:|---|---|
| `3ENTROPY` = 4.24 | 4.57954 | 0.339542 | 4.24e-09 → **fails** | 0.424 → would pass |
| `3ROBUST_MEAN_ABSOLUTE_DEVIATION` = 392.98 | 406.982 | 14.0017 | 1.9649 → **fails** | 39.298 → would pass |

Both failed and named the feature; the values were then restored and all 36 cases pass. The run also
confirms Nyxus' own output matches the oracle — 4.57954 against 4.5795418896949416, and 406.982
against the 406.9817011068 the binned window produces.

## Provenance, which the file never recorded

SPEC §6.4 makes provenance mandatory and the header says outright that it has none: no MATLAB
version, no config, no generator. `not_covered.md` §C tracks it. This report and the generator
supply it, and SPEC §4 already names this table as the one legacy 3D table predating the rule that
`matlab` means Octave.
