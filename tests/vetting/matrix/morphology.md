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
`DIAMETER_EQUAL_PERIMETER` and the five `EDGE_*` statistics — plus `MASS_DISPLACEMENT` and
`ROI_RADIUS_*`, which have their own producers and their own axes. The rest of the family (hull,
caliper, moments-fit ellipse, fractal) has no rows here yet; its recipes are in `config_recipes.md`
and that gap is stated rather than left to read as covered.

## Three producers, three axes

`MASS_DISPLACEMENT` does **not** inherit a contour-builder cell, and no statement about
`buildWholeSlideContour()` or `buildRegularContour_nontriv()` applies to it. It is computed by
`BasicMorphologyFeatures::calculate()` from the geometric and intensity-weighted centroids, reads no
contour at any setting, and has its own `BasicMorphologyFeatures::osized_calculate()`.

`ROI_RADIUS_*` is a third producer, `RoiRadiusFeature`. It *does* consume the contour, but it is not
`ContourFeature`'s output feature, and it consumes the contour differently — as a set of positions
measured against every ROI pixel, rather than as a walk. That distinction is what makes its
out-of-core cell a separate finding from `PERIMETER`'s rather than the same one: `PERIMETER` diverges
because the two builders *summarise* the contour differently, `ROI_RADIUS_*` because they return
different pixels. Its section is below.

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

## `RoiRadiusFeature` — the axis is the run mode, and the contour it is handed

`ROI_RADIUS_MEAN`, `ROI_RADIUS_MAX` and `ROI_RADIUS_MEDIAN` are statistics of the ROI's inradius
map: every ROI pixel's distance to the nearest contour pixel. The feature reads no setting of its
own, so its only axis is the run mode — which decides which contour builder fills `K` — and, for the
oracle cell, which fixture it is measured on.

| run mode | fixture | verdict | oracle / reason |
|---|---|---|---|
| in-RAM (`calculate`) | disks R = 10, 20, 40 | **VALID** | `skimage` on `morphology.radius_disks`, and `analytic` for `MAX` on the same recipe |
| in-RAM (`calculate`) | 8×8 `shape2d` | **VALID-BUT-PRODUCTION-ONLY** | no tool reproduces it *on this fixture*; recipe `morphology.shape2d_native`, drift guard only |
| out-of-core (`osized_calculate`) | disk64 | **impl-defect** | the two paths disagree; recipe `morphology.disk64_forced_ooc` |
| whole-slide (`SINGLEROI=true`) | — | not reached | `buildWholeSlideContour()` pushes four AABB corners, so the inradius map is a corner artefact; no row claimed and none asserted |

**Why the shape2d cell is production-only and the disk cell is not.** The reference —
`find_boundaries(connectivity=1, mode='inner')` plus a minimum over it — reproduces `MAX` and
`MEDIAN` on a disk to double precision and reproduces nothing on the 8×8 raster. The difference is
`buildRegularContour`, which reports every contour pixel one pixel right and one pixel down of where
it is: on the 8×8 mask the traced contour **is** the skimage inner boundary shifted by (+1, +1), all
18 pixels, and shifting the reference the same way reproduces all three Nyxus values. On a disk
`MAX` and `MEDIAN` are unmoved by that shift because the pixel attaining them moves with it, so they
promote there and `MEAN` does not, at either fixture.

### The out-of-core cell is a defect, measured on the same disk that exposed `PERIMETER`

Both paths take `sqrt(exact_min_sqdist(K))` over the same ROI pixels, so the only input that can
differ is `K`. Measured on `bench_disk64_diagonal_boundary`:

| | in-RAM | out-of-core | |
|---|---|---|---|
| contour pixels | 112 | 112 | same size |
| `ROI_RADIUS_MEAN` | 6.087109772358636 | 7.169174091182726 | **+17.8%** |
| `ROI_RADIUS_MEDIAN` | 5.0 | 6.708203932499369 | **+34%** |
| `ROI_RADIUS_MAX` | 19.026297590440446 | 19.026297590440446 | bit-identical |

Same count, different pixels: `buildRegularContour_nontriv` and `buildRegularContour` do not return
the same 112. `MEAN` and `MEDIAN` rise, so some ROI pixels are farther from the out-of-core contour,
and `MAX` is unchanged, so the pixel attaining it is not among them. No translation of the inner
boundary reproduces the out-of-core numbers — searched over every shift in ±4 — so this is pinned as
a characterization rather than explained as an offset, and the explanation is left to whoever
corrects the builder.

`impl-defect` rather than `INVALID`, for the reason the `PERIMETER` cell already gives: the config is
legitimate and reachable production code, and `CLAUDE.md` requires the two paths to return identical
values. Asserted by `test_2d_ooc_roi_radius_diverges_from_in_ram_regression` in
`test_2d_ooc_regression.py`, which pins both sides *and* the inequality, so a partial fix cannot
pass, and pins `MAX`'s equality so a fix that moves the whole contour cannot break the part that
already works.

**It was invisible for the same reason `PERIMETER`'s was.** `test_2d_ooc_invariant.py` runs
`*ALL_MORPHOLOGY*` — which includes these three — and requires every column to agree, but on a
full-image rectangle, where the two contour builders coincide. The disk is what separates them.

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
