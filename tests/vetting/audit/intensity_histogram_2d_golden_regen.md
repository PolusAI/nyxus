# Regenerating the 2D intensity-histogram goldens

Every number pinned in `tests/test_2d_intensity_histogram_mirp.h` and in the phantom table of
`tests/test_2d_intensity_histogram_analytic.h` comes out of a checked-in generator; nothing there is
transcribed by hand. `test_2d_intensity_histogram_ibsi.h` and the small hand-derived fixtures are
the exceptions and are covered at the end. Run everything offline - CI never invokes a reference
tool.

## 1. Stand the tools up

Conda is enough; no Docker.

```bash
conda create -n nyxus_mirp   -c conda-forge python=3.11 mirp numpy
conda create -n nyxus_oracle -c conda-forge python=3.9  pyradiomics simpleitk numpy
```

The analytic generator needs numpy only, so any of these envs runs it. Record what the solver
resolved, next to the goldens:

```bash
conda run -n nyxus_mirp python -c "from importlib import metadata; print(metadata.version('mirp'))"   # 2.6.0
```

## 2. The fixture

The IBSI digital phantom, read out of `tests/test_data.h` by
`tests/vetting/oracles/ibsi_phantom.py`. There is no separate copy to prepare and none to keep in
sync: the C++ tests read the same arrays through `test_2d_intensity_histogram_common.h`.

All four slices form **one** ROI here - 74 of the 80 pixels are in mask - because the intensity
histogram is built over the whole masked voxel set rather than per slice. That is the one place this
family's fixture differs from the GLCM/GLRLM ones, which aggregate per slice.

```bash
python tests/vetting/oracles/ibsi_phantom.py
# z1: 5x6, 20 masked voxels, in-mask levels [1, 4, 6]   ...
```

## 3. Run the generators

```bash
cd tests/vetting/oracles                     # the generators import ibsi_phantom from here
conda run -n nyxus_mirp   python gen_intensity_histogram_mirp.py
conda run -n nyxus_oracle python gen_intensity_histogram_analytic.py
```

Each prints a paste-ready table body plus a provenance header line. Replace the body of
`intensity_histogram_2d_mirp_ref_vals` / `intensity_histogram_2d_analytic_phantom_ref_vals` with it,
keep the header line, rebuild `runAllTests`, run
`--gtest_filter=*INTENSITY_HISTOGRAM*`.

## 4. The configuration each generator is run at

Recipe `ih.ibsi_fbn` on the Nyxus side: `ibsi=true`, `GREYDEPTH=6`, i.e. fixed bin number with 6
bins. IBSI mode is not optional - with `ibsi=false` the whole `IH_*` family is stripped from the
output.

| | MIRP | analytic |
|---|---|---|
| discretisation | `base_discretisation_method="fixed_bin_number"`, `n_bins=6` | `floor(6·(v-lo)/(hi-lo)) + 1`, clipped to [1,6] |
| aggregation | `by_slice=False` (one ROI over all four slices) | one flat voxel list over all four slices |
| domain | bin indices and bin counts only | bin centres, interpolated percentiles, raw intensities |

## 5. Mapping tool names to Nyxus features

MIRP suffixes every column with the discretisation (`ih_mean_fbn_n6`), and one call emits the
GLCM/GLRLM/GLSZM/NGTDM families alongside. The mapping lives in `MIRP_TO_NYXUS` in the generator,
which filters on that suffix so a bin-count change cannot silently read the wrong column:

| Nyxus | MIRP |
|---|---|
| `IH_MEAN_IDX` / `IH_VARIANCE_IDX` | `ih_mean` / `ih_var` |
| `IH_SKEWNESS_IDX` / `IH_EXCESS_KURTOSIS_IDX` | `ih_skew` / `ih_kurt` |
| `IH_MEDIAN_IDX` / `IH_MODE_IDX` | `ih_median` / `ih_mode` |
| `IH_MINIMUM_IDX` / `IH_MAXIMUM_IDX` / `IH_RANGE_IDX` | `ih_min` / `ih_max` / `ih_range` |
| `IH_P10_IDX` / `IH_P90_IDX` / `IH_INTERQUANTILE_RANGE_IDX` | `ih_p10` / `ih_p90` / `ih_iqr` |
| `IH_MEAN_ABSOLUTE_DEVIATION_IDX` | `ih_mad` |
| `IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX` | `ih_rmad` |
| `IH_MEDIAN_ABSOLUTE_DEVIATION_IDX` | `ih_medad` |
| `IH_COEFFICIENT_OF_VARIATION_IDX` | `ih_cov` |
| `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX` | `ih_qcod` |
| `IH_ENTROPY_IDX` / `IH_UNIFORMITY_IDX` | `ih_entropy` / `ih_uniformity` |
| `IH_MAX_GRADIENT` / `IH_MAX_GRADIENT_IDX` | `ih_max_grad` / `ih_max_grad_g` |
| `IH_MIN_GRADIENT` / `IH_MIN_GRADIENT_IDX` | `ih_min_grad` / `ih_min_grad_g` |
| `IH_ROBUST_MEAN_IDX`, the whole `_VAL` family | *(no counterpart - analytic)* |

## 6. Convention differences to expect

- **`_VAL` is not one transform of `_IDX`.** Five conventions coexist; the table is in
  `intensity_histogram_2d_analytic_vetting_report.md`. Do not "simplify" the analytic generator by
  deriving `_VAL` from `_IDX` - four features have no such map, and writing one would make the
  oracle circular.
- **Percentiles are drift guards, not vetted.** `_IDX` is the discrete grey-level percentile, which
  MIRP and both tools' native functions reproduce exactly. `_VAL` is the grouped-data percentile,
  which none of them reproduce - so `IH_P10_VAL`, `IH_P90_VAL`, `IH_INTERQUANTILE_RANGE_VAL` and
  `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` are pinned in
  `tests/test_2d_intensity_histogram_regression.h` with no oracle claim. If you regenerate them,
  update that file, not the analytic one, and do not re-label them `vetted` on the strength of the
  generator agreeing - the generator implements the same method Nyxus does, which is exactly why
  they were demoted.
- **Checking a percentile against a tool's *native* function** (the test that produced the
  demotion): `quantile(v, p, 1, method)` in Octave for methods 1..9 - the third argument is `DIM`,
  so the obvious three-argument `quantile(v, p, method)` silently computes along a dimension and
  returns nonsense for most method numbers. `prctile` is method 5. numpy's equivalent is
  `np.percentile(v, 100*p, method=...)` over the nine documented method names.
- **Index base.** Every `IH_*_IDX` feature is reported 1-based, which is what IBSI quotes and what
  MIRP returns, so no offset is applied anywhere in the generators. `IH_ROBUST_MEAN_IDX` was the one
  exception until this branch; see the analytic report.
- **Gradients.** `IH_MAX_GRADIENT` / `IH_MIN_GRADIENT` are magnitudes over the bin-count curve and
  carry no domain, so they sit in the MIRP table despite having no `_IDX`/`_VAL` pair.

## 7. The goldens no generator produces

- **`test_2d_intensity_histogram_ibsi.h`** - published IBSI consensus (reference manual, "dig
  phantom", intensity-histogram family), not a tool run, so there is nothing to regenerate. To
  re-verify, run MIRP on the phantom at the configuration in section 4 and compare;
  `intensity_histogram_2d_ibsi_vetting_report.md` is the record of doing exactly that.
- **The 17-px tail-trimming fixture** in `test_2d_intensity_histogram_analytic.h` - a fixture chosen
  so the robust window strictly trims both tail bins, which the phantom does not do at the low end.
  Its goldens are derived from the bin counts in a comment above the assertions. If you change the
  fixture, redo that derivation; do not read the new values off a Nyxus run, which is how its
  `IH_ROBUST_MEAN_IDX` golden came to encode a defect.
- **The 5-px `HISTOGRAM` fixture** in the same file - raw per-bin counts `[2,1,2]` for
  `{1,1,3,5,7}` at 3 bins, derived in the comment above the test from the binning contract in
  `src/nyx/features/histogram.h`.
