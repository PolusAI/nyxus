# Audit: `test_2d_firstorder_ibsi.h` goldens vs. the IBSI digital phantom

**Verdict: all 15 hardcoded goldens reproduce exactly (to the file's own 3-sig-fig rounding).**
No drift, no oracle-mismatch.

## Method

Unlike the matlab/pyradiomics audits, no external tool run was needed: the reference is the
published IBSI digital phantom (a fixed, standard benchmark image), and the phantom's own pixel
data is already checked into the repo (`ibsi_phantom_z1..z4_intensity` / `_mask`,
`tests/test_data.h`). Recomputing the statistics directly from that data is the vetting check
itself - no Docker, no Octave, nothing to install.

- Combined the 4 intensity slices with their masks (mask=0 pixels dropped, matching
  `load_masked_test_roi_data` / `assert_firstorder_feature_ibsi` in the test file) -> 74 masked
  voxels out of 80.
- Computed each statistic directly with population-moment conventions (N-denominator, matching
  what the file's own comment implies and what IBSI's reference manual uses) and linear-interpolation
  percentiles.

## Result table

| Feature | Hardcoded (`firstorder_2d_ibsi_ref_vals`) | Recomputed from phantom | Match |
|---|---|---|---|
| MEAN | 2.15 | 2.14865 | ✅ |
| VARIANCE | 3.05 | 3.04547 | ✅ |
| SKEWNESS | 1.08 | 1.08382 | ✅ |
| EXCESS_KURTOSIS | -0.355 | -0.35462 | ✅ |
| MEDIAN | 1 | 1.0 | ✅ |
| MINIMUM | 1 | 1.0 | ✅ |
| P10 | 1 | 1.0 | ✅ |
| P90 | 4 | 4.0 | ✅ |
| MAXIMUM | 6 | 6.0 | ✅ |
| INTERQUARTILE | 3 | 3.0 | ✅ |
| RANGE | 5 | 5.0 | ✅ |
| MEAN_ABSOLUTE_DEVIATION | 1.55 | 1.55223 | ✅ |
| ROBUST_MEAN_ABSOLUTE_DEVIATION | 1.11 | 1.11383 | ✅ |
| ENERGY | 567 | 567.0 | ✅ exact |
| ROOT_MEAN_SQUARED | 2.77 | 2.76806 | ✅ |

## Notes

- `ROBUST_MEAN_ABSOLUTE_DEVIATION` confirms the golden (1.11) is correct. The test asserting it
  (`test_2d_firstorder_robust_mean_absolute_deviation_ibsi`) is commented out in the source with
  "needs to be updated to pass" - since the reference value itself checks out, that's a bug in
  Nyxus's live `ROBUST_MEAN_ABSOLUTE_DEVIATION` calculation (or a settings mismatch in how the test
  invokes it), not a bad golden. See `firstorder_2d_coverage.csv`'s ROBUST_MEAN_ABSOLUTE_DEVIATION
  row for the related pyradiomics-side finding (that one's exact and live).
- No provenance gap here unlike the matlab file: the IBSI phantom and its published reference
  statistics are a fixed external standard, not something generated on demand, so there's no
  version/config/generator-script to pin - the phantom data itself, checked into
  `tests/test_data.h`, is the complete record.
