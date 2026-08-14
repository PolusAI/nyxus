# Audit: `test_2d_intensity_histogram_ibsi.h` goldens vs the IBSI digital phantom

**Verdict: all 12 hardcoded goldens reproduce to their own 3-significant-figure rounding**, and an
independent tool recomputes every one of them from the phantom pixels checked into this repo - so
the "IBSI" label is the published consensus, not Nyxus output relabelled.

## Method

The reference is a published table rather than a tool run, so the risk is circularity: a golden
could have been copied from Nyxus' own output. To rule that out, the phantom stored in
`tests/test_data.h` (`ibsi_phantom_z1..z4_intensity` / `_mask`) was fed to **MIRP 2.6.0** at the
IBSI fixed-bin-number configuration, 6 bins, all four slices as one ROI (see
`intensity_histogram_2d_mirp_vetting_report.md` for the exact settings), and its full-precision
output compared against both the 3-s.f. goldens and Nyxus.

This family gets one corroborating tool rather than the two the GLCM and GLRLM audits used.
PyRadiomics' `firstorder` class reports its statistics in intensity units - its `Mean`, `Median` and
percentiles answer the `_VAL` question, not the `_IDX` one - and the only two that are computed off
the discretised histogram, `Entropy` and `Uniformity`, are exactly the two that are domain-invariant
here, so they take the same value in either family. A PyRadiomics run would therefore corroborate 2
of the 23 quantities and restate them in a domain where MIRP already agrees. MIRP, which works
entirely in the discretised domain, covers all 23.

## Result table

| feature | published golden (3 s.f.) | MIRP fresh | Nyxus |
|---|---|---|---|
| `IH_VARIANCE_IDX` | 3.05 | 3.045471147 | 3.045471147 |
| `IH_SKEWNESS_IDX` | 1.08 | 1.083820723 | 1.083820723 |
| `IH_EXCESS_KURTOSIS_IDX` | -0.355 | -0.3546204807 | -0.3546204807 |
| `IH_INTERQUANTILE_RANGE_IDX` | 3 | 3 | 3 |
| `IH_RANGE_IDX` | 5 | 5 | 5 |
| `IH_MEAN_ABSOLUTE_DEVIATION_IDX` | 1.55 | 1.552227904 | 1.552227904 |
| `IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX` | 1.11 | 1.113833816 | 1.113833816 |
| `IH_MEDIAN_ABSOLUTE_DEVIATION_IDX` | 1.15 | 1.148648649 | 1.148648649 |
| `IH_COEFFICIENT_OF_VARIATION_IDX` | 0.812 | 0.8121978585 | 0.8121978585 |
| `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX` | 0.6 | 0.6 | 0.6 |
| `IH_ENTROPY_IDX` | 1.27 | 1.265611556 | 1.265611556 |
| `IH_UNIFORMITY_IDX` | 0.512 | 0.5124178232 | 0.5124178232 |

Every golden matches MIRP at its own rounding, and Nyxus matches MIRP to double precision on all
twelve. The file's 1% band is loose against that, but the goldens carry only 3 significant figures,
so a tighter band would assert precision the reference does not have. The tight comparison is the
job of `test_2d_intensity_histogram_mirp.h`, which pins the same twelve quantities - and eleven more
- at full precision.

`IH_ROBUST_MEAN_IDX` has no IBSI feature and so appears in no table here; see
`intensity_histogram_2d_analytic_vetting_report.md`, which is also where the defect found in it is
written up.

## The four `_VAL` anchors in this file survive the convention correction

Besides the 12 consensus goldens the file asserts four `_VAL` features against an IBSI-vetted `_IDX`
partner rather than against a golden: `MEAN_ABSOLUTE_DEVIATION_VAL`,
`ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL` and `MEDIAN_ABSOLUTE_DEVIATION_VAL` as `binWidth · IDX`, and
`COEFFICIENT_OF_VARIATION_VAL` as `binWidth · sqrt(VARIANCE_IDX) / MEAN_VAL`.

The analytic report retires the general claim that `_VAL` is an affine image of `_IDX`, so these
four were re-measured rather than assumed. They hold: all three deviation measures are pure-scale
statistics, where the centre map's offset cancels in the difference and only `binWidth` survives
(measured to 1.2e-16), and the CoV anchor does not go through `COEFFICIENT_OF_VARIATION_IDX` at all
- it rebuilds the ratio from `VARIANCE_IDX` and `MEAN_VAL` explicitly, which is why it is one of the
four that works. The features the retired claim got wrong - VARIANCE, the percentiles, MIN/MAX/RANGE
- are not anchored in this file.

## What changed in the file

- It carried its own copy of the 4-slice ROI assembly. Both it and the two new oracle files now go
  through `calc_2d_intensity_histogram_phantom` in `test_2d_intensity_histogram_common.h`, so the
  IBSI consensus, the MIRP goldens and the analytic closed forms are read off one computation at one
  configuration.
- `IH_PHANTOM_NBINS` moved to that shared header with it. The value is unchanged.
- No golden in this file changed.
