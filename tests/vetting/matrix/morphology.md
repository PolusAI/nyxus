# 2D morphology config matrix — contour and edge-intensity cells

Axes = the settings the feature actually reads; verdicts are **measured**, per SPEC §5.2, not
hand-labelled. The three run-mode cells are measured on `bench_disk64_diagonal_boundary`
(`benchmarks.md`), built by `tests/python/test_data.py::disk64_arrays` so that every module reading
it reads one definition. The CellProfiler oracle cell for the `EDGE_*` statistics is measured on
`bench_shape8_concave_holed`, which is the fixture its recipe pins.

**Dispositions are SPEC §5.1's, used literally.** `VALID` means *an external tool computes this and
agrees* — oracle-backed, nothing else. `VALID-BUT-PRODUCTION-ONLY` means a real config no tool
reproduces, kept as a snapshot. `INVALID` means degenerate or nonsensical, dropped with a reason.
Two cells here are none of those and are labelled accordingly:

- **`INVARIANT`** — backed by a required relation between two Nyxus code paths, with no tool
  involved. SPEC §3 lists `invariant` among the outcome values; it is not a weaker `vetted`, and
  reading path-equality as evidence of correctness is exactly the mistake it exists to prevent.
- **`impl-defect`** — real production code whose two implementations are required to agree and do
  not. That is a defect, not an `INVALID` config: the configuration is legitimate and reachable, so
  it cannot be dropped with a reason.

**Scope.** This file covers the features `ContourFeature` produces — `PERIMETER`,
`DIAMETER_EQUAL_PERIMETER` and the five `EDGE_*` statistics — plus `MASS_DISPLACEMENT`, which has
its own producer and its own axis. The rest of the family (hull, caliper, moments-fit ellipse,
fractal) has no rows here yet; its recipes are in `config_recipes.md` and that gap is stated rather
than left to read as covered.

## Two producers, two axes

`MASS_DISPLACEMENT` does **not** inherit a contour-builder cell, and no statement about
`buildWholeSlideContour()` or `buildRegularContour_nontriv()` applies to it. It is computed by
`BasicMorphologyFeatures::calculate()` from the geometric and intensity-weighted centroids, reads no
contour at any setting, and has its own `BasicMorphologyFeatures::osized_calculate()`.

### `ContourFeature` — the axis is which contour builder runs

| cell | path | `EDGE_*` | `PERIMETER`, `DIAMETER_EQUAL_PERIMETER` |
|---|---|---|---|
| segmented, in-RAM | `calculate()` at `SINGLEROI=false` → `buildRegularContour()` | **VALID** — cellprofiler, recipe `morphology.cellprofiler_edge_intensity` (shape8 fixture) | **VALID** — skimage, recipe `morphology.perimeter_circles` |
| forced out-of-core | `osized_calculate()` → `buildRegularContour_nontriv()` | **INVARIANT** — measured equal to the in-RAM cell | **impl-defect** — measured *unequal*, see below |
| whole-slide | `calculate()` at `SINGLEROI=true` → `buildWholeSlideContour()` | **VALID-BUT-PRODUCTION-ONLY** — degenerate, and CellProfiler measured as not comparable | **VALID-BUT-PRODUCTION-ONLY** — AABB walk |

### `BasicMorphologyFeatures` — `MASS_DISPLACEMENT`

| cell | path | verdict |
|---|---|---|
| segmented, in-RAM | `calculate()` | **VALID** — cellprofiler, recipe `morphology.cellprofiler_edge_intensity` |
| forced out-of-core | `osized_calculate()` | **INVARIANT** — measured equal to the in-RAM cell |
| whole-slide | `calculate()`, ROI = the whole frame | **VALID** — cellprofiler, recipe `morphology.cellprofiler_wholeslide_massdisp`, `rel=7e-9` |

That last cell is the one the run changed. It was provisionally written production-only on the
assumption that whole-slide mode is not comparable; running CellProfiler with one object over the
full frame showed it **is** — 3.345311793150965 against Nyxus' 3.3453118163885427 — because this
feature never touches the contour and is the same quantity whether the object is a disk or a frame.
Its `EDGE_*` siblings at the same cell are *not*: CellProfiler returns exactly 0 for each, because
an all-ones label image has an empty `find_boundaries(mode="inner")` set. A tool that emits a number
under a matching name is not thereby an oracle for it, and
`oracles/gen_morphology_wholeslide_cellprofiler.py` asserts that zero rather than describing it.

## The measurements

| feature | segmented in-RAM | forced out-of-core | whole-slide |
|---|---:|---:|---:|
| `PERIMETER` | 131.88225099390849 | **112.0** | 256.0 |
| `DIAMETER_EQUAL_PERIMETER` | 41.97942430353313 | **35.65070725258456** | 81.48733086305042 |
| `MASS_DISPLACEMENT` | 2.7526140113386943 | 2.7526140113386943 | 3.3453118163885427 |
| `EDGE_MEAN_INTENSITY` | 257.0 | 257.0 | 397.0 |
| `EDGE_STDDEV_INTENSITY` | 98.12659593009853 | 98.12659593009853 | 0.0 |
| `EDGE_MAX_INTENSITY` | 397.0 | 397.0 | 397.0 |
| `EDGE_MIN_INTENSITY` | 117.0 | 117.0 | 397.0 |
| `EDGE_INTEGRATED_INTENSITY` | 28784.0 | 28784.0 | 1588.0 |

**What backs each column, because they are not backed the same way.** The segmented and whole-slide
columns are pinned literally, in `SEGMENTED` and `WHOLE_SLIDE` in
`tests/python/test_2d_morphology_regression.py`. In the out-of-core column only `PERIMETER` and
`DIAMETER_EQUAL_PERIMETER` are pinned to literals (`test_2d_ooc_regression.py`); the other six cells
are asserted **equal to the segmented column** by
`test_2d_ooc_2d_contour_intensity_matches_in_ram_on_diagonal_boundary_invariant`, not pinned
independently — so a change that moved both paths together would keep those six agreeing while the
segmented pin caught it. Every cell has its own registry row naming its recipe.

## The out-of-core contour divergence is a defect, not a convention

`calculate()` sums Euclidean step lengths around the contour; `osized_calculate()` sets
`fval_PERIMETER = (StatsInt) K.size()`, the contour pixel **count**. On the disk that is 131.882
against 112.0 — and 112 is exactly the fixture's edge-pixel count (`benchmarks.md`), which
identifies the cause as the definition rather than an accumulation error.

**`DIAMETER_EQUAL_PERIMETER` inherits it exactly.** Both paths compute it as `fval_PERIMETER / M_PI`
(`contour.cpp` lines 976 and 1000), so the ratio between the two paths is identical for the two
features: 41.979 against 35.651. It is one defect with two public consequences, not two defects —
`test_2d_ooc_regression.py` asserts that identity, so a fix to `PERIMETER` alone must fix this one
too, and a future run where the two ratios stop matching is a new finding rather than this one.

`CLAUDE.md` requires that "the in-RAM path and the out-of-core path must produce identical values",
so this is a defect. It is characterized rather than fixed here: the fix is a `src/nyx` change, and
`PERIMETER` is vetted under a different recipe. Both rows carry `flag=impl-defect` — **not** a
dropped `INVALID` cell, because the configuration is real production code that runs whenever an ROI
exceeds `ram_limit`.

**Why no existing test caught it.** `test_2d_ooc_invariant.py` has asserted `*ALL_MORPHOLOGY*`
equality across the two paths since it was written, and passes. Its fixture is a full-image
**rectangle**, and around a rectangle every contour step is an axis-aligned unit step — so the pixel
count and the Euclidean sum are the same number, and the two features that differ cannot appear. The
invariant is sound; that shape cannot discriminate. The general form is recorded in
`not_covered.md` §G: *a path-equality fixture symmetric in the axis under test proves nothing about
that axis.*

## The whole-slide cell is degenerate by construction

`buildWholeSlideContour()` does not trace a boundary: it pushes the four AABB corners, each carrying
`r.aux_max`. So `EDGE_MIN == EDGE_MAX == EDGE_MEAN ==` the image maximum, `EDGE_STDDEV == 0`, and
`EDGE_INTEGRATED == 4 × maximum` — confirmed by the table above (397, 397, 397, 0, 1588) — while
`PERIMETER` is the AABB walk. These are statistics of four synthesised points, not of an object's
edge, which is why the `vetted` `EDGE_*` rows state their scope as the segmented in-RAM path and why
`test_2d_morphology_regression.py` asserts that the two cells actually disagree.

## EDGE_STDDEV_INTENSITY is off the oracle axis by definition

At every cell above, Nyxus divides the variance by n-1 (`Moments4::std()`) and CellProfiler by n, so
the two differ by exactly `sqrt(n/(n-1))` — 2.9% at n=18 on the oracle fixture — regardless of which
contour builder produced the pixel set. That is a definitional gap rather than a config point, so
the feature is never `VALID` at any cell. See `not_covered.md` §C.
