# 2D morphology config matrix — contour and edge-intensity cells

Axes = the settings the feature actually reads; verdicts are **measured**, per SPEC §5.2, not
hand-labelled. The measurements below were taken on a 64×64 single-disk fixture (intensity
`1 + x + 7y` inside the disk, maximum 397), built identically by
`tests/python/test_2d_ooc_invariant.py`, `test_2d_ooc_regression.py` and
`test_2d_morphology_regression.py` so all three cells are comparable numbers on one shape. The
CellProfiler oracle cell is measured on `bench_shape8_concave_holed` instead, which is the fixture
its recipe pins.

**Scope.** This file covers the features produced by `ContourFeature` — `PERIMETER`,
`DIAMETER_EQUAL_PERIMETER`, the five `EDGE_*` statistics — plus `MASS_DISPLACEMENT`, which is
produced by `BasicMorphologyFeatures` but asserted by the same oracle test. The rest of the family
(hull, caliper, moments-fit ellipse, fractal) has no rows here yet; its recipes are in
`config_recipes.md` and that gap is stated rather than left to read as covered.

## Two feature groups, two production axes

`MASS_DISPLACEMENT` does **not** inherit a contour-builder cell. It is computed by
`BasicMorphologyFeatures::calculate()` from the geometric and intensity-weighted centroids and never
reads a contour, and it has its own `BasicMorphologyFeatures::osized_calculate()`. It therefore gets
its own axis below, which happens to land on the same dispositions — but by measurement, not by
inheritance.

### `ContourFeature` — the axis is which contour builder runs

| cell | path | EDGE_* verdict | PERIMETER verdict |
|---|---|---|---|
| segmented, in-RAM | `calculate()` at `SINGLEROI=false` → `buildRegularContour()` | **VALID** — cellprofiler, recipe `morphology.cellprofiler_edge_intensity` | **VALID** — skimage, recipe `morphology.perimeter_circles` |
| out-of-core | `osized_calculate()` → `buildRegularContour_nontriv()` | **VALID** — equals the in-RAM cell exactly (invariant, measured) | **INVALID — defect**, see below |
| whole-slide | `calculate()` at `SINGLEROI=true` → `buildWholeSlideContour()` | **VALID-BUT-PRODUCTION-ONLY** — degenerate, snapshot | **VALID-BUT-PRODUCTION-ONLY** — AABB walk, snapshot |

### `BasicMorphologyFeatures` — `MASS_DISPLACEMENT`

| cell | path | verdict |
|---|---|---|
| segmented, in-RAM | `calculate()` | **VALID** — cellprofiler, recipe `morphology.cellprofiler_edge_intensity` |
| out-of-core | `osized_calculate()` | **VALID** — equals the in-RAM cell exactly (2.7526140113386943 both sides) |
| whole-slide | `calculate()`, ROI = whole image | **VALID-BUT-PRODUCTION-ONLY** — 3.3453118163885427, snapshot |

## The measurements

| feature | segmented in-RAM | out-of-core | whole-slide |
|---|---:|---:|---:|
| `PERIMETER` | 131.88225099390849 | **112.0** | 256.0 |
| `MASS_DISPLACEMENT` | 2.7526140113386943 | 2.7526140113386943 | 3.3453118163885427 |
| `EDGE_MEAN_INTENSITY` | 257.0 | 257.0 | 397.0 |
| `EDGE_STDDEV_INTENSITY` | 98.12659593009853 | 98.12659593009853 | 0.0 |
| `EDGE_MAX_INTENSITY` | 397.0 | 397.0 | 397.0 |
| `EDGE_MIN_INTENSITY` | 117.0 | 117.0 | 397.0 |
| `EDGE_INTEGRATED_INTENSITY` | 28784.0 | 28784.0 | 1588.0 |

Assertions: `test_2d_ooc_invariant.py`
(`test_2d_ooc_2d_contour_intensity_matches_in_ram_on_diagonal_boundary_invariant`),
`test_2d_ooc_regression.py`, `test_2d_morphology_regression.py`, and for the oracle cell
`test_2d_morphology_cellprofiler.h`.

## The out-of-core PERIMETER divergence is a defect, not a convention

`calculate()` sums Euclidean step lengths around the contour; `osized_calculate()` sets
`fval_PERIMETER = (StatsInt) K.size()`, the contour pixel **count**. On the disk that is 131.882
against 112.0 — a 15% divergence, and the out-of-core value is an integer, which is what identifies
the cause as the definition rather than an accumulation error. `CLAUDE.md` requires that "the in-RAM
path and the out-of-core path must produce identical values", so this is a defect. It is
characterized rather than fixed here: the fix is a `src/nyx` change and `PERIMETER` is vetted under
a different recipe. Pinned by `test_2d_ooc_regression.py`, which a correct fix must break.

**Why no existing test caught it.** `test_2d_ooc_invariant.py` has asserted `*ALL_MORPHOLOGY*`
equality across the two paths since it was written, and passes. Its fixture is a full-image
**rectangle**, and around a rectangle every contour step is an axis-aligned unit step, so the pixel
count and the Euclidean sum are the same number. The invariant is real; that shape simply cannot
discriminate. The disk fixture separates them, which is why the new assertions use it.

## The whole-slide cell is degenerate by construction

`buildWholeSlideContour()` does not trace a boundary: it pushes the four AABB corners, each carrying
`r.aux_max`. So `EDGE_MIN == EDGE_MAX == EDGE_MEAN ==` the image maximum, `EDGE_STDDEV == 0`, and
`EDGE_INTEGRATED == 4 × maximum` — confirmed by the table above (397, 397, 397, 0, 1588). These are
statistics of four synthetic points, not of an object's edge, so no external tool reproduces them
and the cell can only ever be a snapshot. That is also why the `vetted` rows for these features
state their scope as the segmented in-RAM path: the CellProfiler evidence does not carry across this
cell, and `test_2d_morphology_regression.py` asserts the two cells actually disagree so that nobody
reads it as though it did.

## EDGE_STDDEV_INTENSITY is off the oracle axis by definition

At every cell above, Nyxus divides the variance by n-1 (`Moments4::std()`) and CellProfiler by n, so
the two differ by exactly `sqrt(n/(n-1))` — 2.9% at n=18 on the oracle fixture — regardless of which
contour builder produced the pixel set. That is a definitional gap rather than a config point, so
the feature stays `regression` at all three cells and no recipe promotes it. See `not_covered.md`
§C.
