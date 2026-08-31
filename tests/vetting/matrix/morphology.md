# 2D morphology config matrix — contour and edge-intensity cells

Axes = the settings the feature actually reads; verdicts are measured on
`bench_shape8_concave_holed` unless a cell says otherwise (SPEC §5.1).

**Scope.** This file currently covers the `ContourFeature` group only — `PERIMETER`,
`DIAMETER_EQUAL_PERIMETER` and the five `EDGE_*` edge-intensity statistics — because that is the
group the CellProfiler recipe makes a `vetted` claim about, and SPEC §5.1 asks a recipe to name the
production cell it covers. The rest of the family (hull, caliper, moments-fit ellipse, fractal) has
no rows here yet; its recipes are in `config_recipes.md` and the gap is real rather than implied
covered. `MASS_DISPLACEMENT` sits in `basic_morphology` rather than `ContourFeature` but is asserted
by the same test at the same settings, so it inherits this cell.

## The axis is which contour builder runs, not a numeric knob

The edge statistics are all computed by one helper,
`calc_min_max_mean_stddev_intensity(K)` in `contour.cpp`, over the flattened contour `K` that
`LR::merge_multicontour` produces. So the arithmetic is shared across every path and the only thing
that varies is **which pixels land in `K`**. Three code paths build it:

| cell | path | verdict |
|---|---|---|
| segmented, in-RAM | `calculate()` at `SINGLEROI=false` → `buildRegularContour()` | **VALID — vetted**, `morphology.cellprofiler_edge_intensity` |
| whole-slide | `calculate()` at `SINGLEROI=true` → `buildWholeSlideContour()` | NOT MEASURED |
| out-of-core | `osized_calculate()` → `buildRegularContour_nontriv()` | NOT MEASURED |

## Why the segmented in-RAM cell is the one measured

It is the path every 2D segmented run takes and the one the gtest fixture exercises:
`make_shape2d_settings()` sets `SINGLEROI=false`, so `calculate()` dispatches to
`buildRegularContour()`. It is also the only cell where a CellProfiler comparison is a
same-quantity comparison — CellProfiler's `MeasureObjectIntensity` measures per labelled object, so
it has nothing to say about the whole-slide contour, which is not a per-object boundary at all.

At this cell the ROI is concave with an interior hole, so `merge_multicontour` concatenates two
contours and the edge set is 18 of 26 pixels rather than a trivial ring — the statistics are
non-degenerate in both the min/max and the spread.

## Not measured

**Whole-slide (`SINGLEROI=true`).** `buildWholeSlideContour()` does not trace a boundary at all: it
pushes the four AABB corners, each carrying `r.aux_max` as its intensity. Every edge statistic over
that `K` is therefore degenerate by construction — min, max and mean are all the slide maximum, the
stddev is 0, and the integrated value is four times the maximum — and none of them is the
per-object edge quantity CellProfiler measures. This is a different quantity rather than the same
one at another setting, so it needs its own row and its own oracle rather than inheriting this
cell's evidence. Nothing in the tree asserts these features at that setting today.

**Out-of-core (`osized_calculate`).** Reachable for ROIs too large to hold in RAM, and it feeds the
same `calc_min_max_mean_stddev_intensity` helper — but from `buildRegularContour_nontriv()`, a
second contour construction, so equality with the in-RAM cell is an assumption rather than a
measurement. Two differences are visible in the source and are recorded here as open points, not as
findings, because this vetting did not run that path:

- `osized_calculate()` sets `fval_PERIMETER = (StatsInt) K.size()`, the contour pixel **count**,
  where `calculate()` sums the Euclidean step lengths around `K`. On this fixture those are 26.935
  and a whole number respectively, so the two paths do not agree on `PERIMETER` by construction.
- `calculate()` guards an empty contour by zeroing all seven values; `osized_calculate()` has no
  such branch.

Neither is asserted anywhere, so both are untriaged rather than ruled out. The in-RAM and
out-of-core paths are required to produce identical values (`CLAUDE.md`, "the in-RAM path and the
out-of-core path must produce identical values"), which makes this a cell worth measuring rather
than a documented divergence.

## EDGE_STDDEV_INTENSITY is off the oracle axis by definition

At every cell above, Nyxus divides the variance by n-1 (`Moments4::std()`) and CellProfiler by n,
so the two differ by exactly `sqrt(n/(n-1))` — 2.9% at n=18 — regardless of which contour builder
produced `K`. That is a definitional gap rather than a config point, so the feature stays
`regression` at all three cells and no recipe promotes it. See `not_covered.md` §C.
