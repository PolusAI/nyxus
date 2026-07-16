# Bug: Intensity-Histogram IQR/QCoD `_IDX` variants floor quantiles while `_VAL` interpolate

**Component:** `src/nyx/features/intensity_histogram.cpp` — `IntensityHistogramFeatures`
**Severity:** low (internal inconsistency; affects the `_IDX` quantile-derived variants)
**Found:** 2026-07-16, during IBSI oracle-vetting of the IH dispersion/index family.

## Summary

For the intensity-histogram quantile-based features, the `_VAL` (bin-center value) and `_IDX`
(bin-index) variants are **not** consistent representations of the same statistic. `_VAL` uses the
interpolated within-bin quantile; `_IDX` uses the *floored* bin index of that quantile. As a result
`IQR_VAL ≠ binWidth · IQR_IDX`, even though for every non-quantile dispersion feature (mean absolute
deviation, robust MAD, median AD, variance, …) the `_VAL = binWidth · _IDX` identity holds exactly.

## Root cause

In `IntensityHistogramFeatures::calculate` (`intensity_histogram.cpp`):

- `p25Value = quantile(0.25)`, `p75Value = quantile(0.75)` — the `quantile` lambda interpolates
  *within* the bin (continuous).
- `p25Index = getIndexOf(p25Value)`, `p75Index = getIndexOf(p75Value)` — `getIndexOf` returns an
  `int` bin index, i.e. it **floors** the continuous quantile, discarding the sub-bin fraction.
- `IH_INTERQUANTILE_RANGE_VAL = p75Value − p25Value` (interpolated),
  `IH_INTERQUANTILE_RANGE_IDX = p75Index − p25Index` (floored).
- Same split for `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_{VAL,IDX}`.

The mean/variance/MAD-family features instead compute the `_IDX` variant directly in the per-bin loop
using the unfloored loop counter `i`, so their `_VAL`/`_IDX` pair stays affine-consistent.

## Reproduction (IBSI digital phantom, GREYDEPTH=6, IBSI mode)

`binWidth ≈ 0.8333`, `IH_INTERQUANTILE_RANGE_IDX = 3`.

- Expected under the affine identity: `IQR_VAL = 0.8333 · 3 = 2.5`.
- Actual: `IH_INTERQUANTILE_RANGE_VAL = 2.42604` (~3% off — far beyond floating-point).

## Impact

- The IBSI-vetted `_IDX` quantile features are unaffected and correct against IBSI consensus
  (`IQR_IDX = 3`, `QCoD_IDX = 0.6`).
- Only the cross-domain identity is broken. In the oracle-vetting work, `IQR_VAL` and `QCoD_VAL`
  therefore cannot be transform-anchored to their IBSI-vetted `_IDX` values; they are vetted
  analytically instead.

## Suggested fix (out of scope for the test-only vetting branch)

Make `_IDX` quantile features use the continuous (interpolated) quantile position rather than the
floored bin index — i.e. compute `IQR_IDX`/`QCoD_IDX` from the same interpolated quantiles as the
`_VAL` variants, expressed in index units — so the `_VAL = binWidth · _IDX` identity holds for the
quantile family too. Decide deliberately whether `_IDX` should be integer-index or continuous; either
way, the two variants should be consistent.
