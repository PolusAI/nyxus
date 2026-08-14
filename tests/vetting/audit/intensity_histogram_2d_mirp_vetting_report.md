# Audit: 2D intensity histogram vs a fresh MIRP run

**Verdict: all 23 quantities of the discretised (`_IDX`) family reproduce to double precision**,
worst 1.3e-15. Nothing in this family is asserted at a loosened band.

Covers `tests/test_2d_intensity_histogram_mirp.h` (goldens + assertions),
`tests/test_2d_intensity_histogram_common.h` (fixture) and
`tests/vetting/oracles/gen_intensity_histogram_mirp.py` (generator).

## Method

- **Tool**: mirp **2.6.0** in a conda env (`conda create -n nyxus_mirp -c conda-forge python=3.11
  mirp numpy`). MIRP is the IBSI reference implementation, so it is the second opinion on the
  published consensus rather than a PyRadiomics-shaped restatement of it.
- **Config**: `base_discretisation_method="fixed_bin_number"`, `base_discretisation_n_bins=6`,
  `by_slice=False`. Nyxus side: recipe `ih.ibsi_fbn` (`ibsi=true`, `GREYDEPTH=6`).
- **Fixture**: the IBSI phantom from `tests/test_data.h` via `ibsi_phantom.py` - all four slices as
  one ROI, 74 of the 80 pixels in mask, which is the voxel set the histogram is built over.
- **Command**: `python tests/vetting/oracles/gen_intensity_histogram_mirp.py`.
- **Gotcha**: MIRP logs at INFO onto stdout, interleaving progress lines with the golden table it
  prints. The generator calls `logging.disable(logging.INFO)`; setting a level on the root logger is
  not enough, because MIRP configures its own logger during the run.
- **Gotcha**: MIRP suffixes every column with the discretisation it was computed at
  (`ih_mean_fbn_n6`), and one run emits the GLCM/GLRLM/GLSZM/NGTDM families too. The generator
  filters on the suffix and maps by name, so a bin-count change cannot silently read the wrong
  column.

## Result table

| feature | Nyxus | fresh run | rel | verdict |
|---|---|---|---:|---|
| `IH_COEFFICIENT_OF_VARIATION_IDX` | 0.8121978584917314 | 0.8121978584917314 | 0.0e+00 | vetted |
| `IH_ENTROPY_IDX` | 1.2656115555865246 | 1.2656115555865246 | 0.0e+00 | vetted |
| `IH_EXCESS_KURTOSIS_IDX` | -0.3546204806878346 | -0.35462048068783414 | 1.3e-15 | vetted |
| `IH_INTERQUANTILE_RANGE_IDX` | 3 | 3 | 0.0e+00 | vetted |
| `IH_MAXIMUM_IDX` | 6 | 6 | 0.0e+00 | vetted |
| `IH_MAX_GRADIENT` | 8 | 8 | 0.0e+00 | vetted |
| `IH_MAX_GRADIENT_IDX` | 3 | 3 | 0.0e+00 | vetted |
| `IH_MEAN_ABSOLUTE_DEVIATION_IDX` | 1.5522279035792552 | 1.552227903579255 | 1.4e-16 | vetted |
| `IH_MEAN_IDX` | 2.1486486486486487 | 2.1486486486486487 | 0.0e+00 | vetted |
| `IH_MEDIAN_ABSOLUTE_DEVIATION_IDX` | 1.1486486486486487 | 1.1486486486486487 | 0.0e+00 | vetted |
| `IH_MEDIAN_IDX` | 1 | 1 | 0.0e+00 | vetted |
| `IH_MINIMUM_IDX` | 1 | 1 | 0.0e+00 | vetted |
| `IH_MIN_GRADIENT` | -50 | -50 | 0.0e+00 | vetted |
| `IH_MIN_GRADIENT_IDX` | 1 | 1 | 0.0e+00 | vetted |
| `IH_MODE_IDX` | 1 | 1 | 0.0e+00 | vetted |
| `IH_P10_IDX` | 1 | 1 | 0.0e+00 | vetted |
| `IH_P90_IDX` | 4 | 4 | 0.0e+00 | vetted |
| `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX` | 0.6 | 0.6 | 0.0e+00 | vetted |
| `IH_RANGE_IDX` | 5 | 5 | 0.0e+00 | vetted |
| `IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX` | 1.1138338159946537 | 1.113833815994654 | 2.0e-16 | vetted |
| `IH_SKEWNESS_IDX` | 1.083820722557456 | 1.0838207225574563 | 2.0e-16 | vetted |
| `IH_UNIFORMITY_IDX` | 0.5124178232286339 | 0.5124178232286339 | 0.0e+00 | vetted |
| `IH_VARIANCE_IDX` | 3.045471146822498 | 3.045471146822498 | 0.0e+00 | vetted |

The Nyxus column is a run of the shipped library on the same 74 voxels (Python API, `ibsi=True`,
`coarse_gray_depth=6`), which reproduces the gtest fixture's histogram exactly.

## What this changes about the family's labelling

31 of these rows previously read `oracle=analytic` or `oracle=ibsi` with `source=audit` - a verdict
reached offline, with no generator in the tree behind it. All 23 `_IDX` quantities turn out to have
a real tool counterpart, so the rows now name `mirp` and point at goldens that a checked-in script
regenerates. **Nothing is demoted**; the values were right, the provenance was not.

## The one feature MIRP does not cover

`IH_ROBUST_MEAN_IDX` - the mean over the [P10,P90]-trimmed histogram - has no MIRP column and no
IBSI counterpart. The test's coverage invariant, which requires every `_IDX` feature the build
exposes to be pinned in the golden table, exempts it by name with that reason; it is vetted
analytically instead. That invariant is the point of the loop: without it a feature added to the
family later would be vetted by nothing while this test still passed over the table it happened to
have.

Running the exemption down is what surfaced a defect - see
`intensity_histogram_2d_analytic_vetting_report.md`.

## What this report does and does not establish

The in-tree goldens were emitted by the generator named above, so "golden == fresh run" only shows
the pin is reproducible. The vetting claim rests on the **Nyxus vs tool** columns: two independent
implementations of the same published definitions, at the same configuration, on the same pixels.
