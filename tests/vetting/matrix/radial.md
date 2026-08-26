# 2D radial config matrix

`RadialDistributionFeature::calculate()` receives `Fsettings` but reads no setting from it. The
number of radial bins and angular wedges is the compile-time constant `num_bins = 8`; the centre,
normalising radius, bin-index rule and three reported quantities are implementation semantics, not
config knobs. The curated cross-product required by SPEC §5 therefore contains one production
point.

| radial bins | runtime settings read | candidate oracle mapping | measured result | verdict / disposition |
|---:|---|---|---|---|
| 8 (compile-time) | none | `radial.cellprofiler_8bin` — CellProfiler `MeasureObjectIntensityDistribution`, scaled 8-bin mode | 21 of 24 feature-bin values differ by more than `rel=1e-2`; the other 3 are bins both tools leave empty | **VALID-BUT-PRODUCTION-ONLY** — recipe `radial.shape2d_native`; keep all three features as regression |

## Why the CellProfiler run is not another config point

The CellProfiler recipe is the candidate oracle mapping for the same single Nyxus point. It does
not create a second Nyxus configuration because there is no radial setting that selects
CellProfiler semantics. The mapping was rejected after measurement because each named feature
computes a different quantity:

- `FRAC_AT_D`: Nyxus pixel-count fraction versus CellProfiler intensity fraction;
- `MEAN_FRAC`: Nyxus raw bin mean versus CellProfiler ROI-mean-normalised value;
- `RADIAL_CV`: Nyxus CV of eight wedge sums including empty wedges versus CellProfiler CV of
  non-empty wedge means.

The centre, radial coordinate and bin-index rules also differ. These are definition and
implementation differences, so no tolerance can promote the point. The complete comparison is in
`../audit/radial_2d_cellprofiler_vetting_report.md`, reproduced by
`../oracles/gen_radial_cellprofiler.py`.

## What is asserted at the production point

`test_2d_radial_regression.h` pins all 24 Nyxus values—three features by eight bins—at `rel=1e-9`.
The written model and a fresh Nyxus build reproduce those pins with measured relative residual 0.
`test_2d_radial_invariant.h` separately checks properties that survive either definition. Neither
kind establishes oracle vetting, so the three registry rows correctly remain `status=regression`
with an empty `oracle` column.

## Why the mechanics tests do not replace this matrix

`test_2d_radial_mechanics.h` characterizes the contour frame, selected centre, normalising radius
and last-bin occupancy that produce the current values. Those assertions pin known defects that a
correction is expected to change; they are diagnostic implementation checks, not config points and
not oracle evidence. The matrix records the family-level disposition that those tests cannot:
there is one real production cell, its only plausible oracle mapping was measured and rejected,
and the cell is therefore regression-only.

Empty ROIs and missing contours are input-degenerate cases handled by an early return in
`radial_distribution.cpp`; they are not settings and therefore are not additional matrix cells.
Likewise, the in-memory and oversized-ROI paths are execution paths for the same point, not config
axes.

## What would change the verdict

Promotion requires resolving the six source-level divergences in the audit report, adding a larger
fixture with a unique distance-to-edge maximum and enough occupancy across eight bins, and rerunning
the CellProfiler mapping. Until then the single matrix cell remains production-only.
