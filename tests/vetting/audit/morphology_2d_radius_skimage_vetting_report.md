# 2D ROI_RADIUS vs scikit-image — vetting report

Vetting record for the three `ROI_RADIUS_*` features. Two of them carry oracle rows; the third does
not, and the last section measures exactly what stands in its way.

## Tool and configuration

| | |
|---|---|
| Tool | scikit-image 0.26.0, scipy 1.17.1, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_morphology_radius_skimage.py` |
| Recipe | `morphology.radius_disks` |
| Benchmark | `bench_radius_disks` — filled digital disks at R = 10, 20, 40 |
| Tests | `test_2d_morphology_skimage.h` (`TEST_2D_MORPHOLOGY_ROI_RADIUS_DISKS_SKIMAGE`) and `test_2d_morphology_analytic.h` (`TEST_2D_MORPHOLOGY_ROI_RADIUS_DISK_CLOSED_FORM_ANALYTIC`) |
| Tolerance | `rel=1e-3` for the skimage pins, `exact` for the closed form |

```
python tests/vetting/oracles/gen_morphology_radius_skimage.py
```

Verifies every golden pinned in the disk table of the test header and exits non-zero on a mismatch,
on an empty table as well as on a wrong value. Current run: **6 verified, 0 failed, 0 unproducible**.

## What the features measure

`RoiRadiusFeature` (`src/nyx/features/roi_radius.cpp`) takes every ROI pixel, measures its distance
to the nearest pixel of the ROI's own contour, and reports the mean, maximum and median of those
distances. Both the in-RAM and the out-of-core path compute it the same way, through
`Pixel2::exact_min_sqdist()` — a full linear scan of the contour, rather than the approximate
hill-descent `min_sqdist()` beside it, which assumes a locally-unimodal ordered contour and can
settle in the wrong basin on a closed one.

## The reference

The same quantity, from two library calls, with no part of Nyxus re-implemented:

```
boundary = skimage.segmentation.find_boundaries(mask, connectivity=1, mode='inner')
radii    = min distance from each mask pixel to a boundary pixel
```

`find_boundaries(connectivity=1, mode='inner')` is the whole of the convention — the foreground
pixels 4-adjacent to background. It is **CellProfiler's own edge definition**, which is why
`gen_morphology_cellprofiler.py` and `gen_morphology_wholeslide_cellprofiler.py` already reproduce
the `EDGE_*` statistics through it. On the 8×8 mask it agrees with the Nyxus chain-code contour pixel
for pixel, up to the offset the last section describes; on the r=20 disk it returns **112** boundary
pixels, the count `benchmarks.md` records for `bench_disk64_diagonal_boundary`, which is the same
1257-pixel shape.

The distance step carries no convention and is computed twice — an exhaustive minimum and
`scipy.ndimage.distance_transform_edt` over the complement of the boundary — with the two required to
agree, so neither is taken on trust.

**Disks and not the 8×8 raster**, for the reason `morphology.perimeter_circles` exists: on 26 pixels
with a hole the two boundary conventions have nothing to converge to. **Three radii and not one**,
because a single disk cannot distinguish a distance from its square — either is just a number on one
fixture, while across a doubling of R a distance grows 2.10× and 2.05× and a square 4.41× and 4.20×.

## Results

| R | pixels | feature | Nyxus | scikit-image | rel |
|---|---|---|---|---|---|
| 10 | 317 | ROI_RADIUS_MAX | 9.0553851381374173 | 9.055385138137417 | 0 |
| 10 | 317 | ROI_RADIUS_MEDIAN | 2.2360679774997898 | 2.23606797749979 | 0 |
| 20 | 1257 | ROI_RADIUS_MAX | 19.026297590440446 | 19.026297590440446 | 0 |
| 20 | 1257 | ROI_RADIUS_MEDIAN | 5.0 | 5.0 | 0 |
| 40 | 5025 | ROI_RADIUS_MAX | 39.012818406262319 | 39.01281840626232 | 0 |
| 40 | 5025 | ROI_RADIUS_MEDIAN | 11.045361017187261 | 11.045361017187261 | 0 |

## A second oracle, for what the pins cannot see

The pins say Nyxus agrees with a reference implementation of the definition. They cannot say the
value obeys the geometry — a reference sharing a mistake would agree with it. So the closed form is a
separate assertion with a separate oracle (`test_2d_morphology_analytic.h`), which also gives
`ROI_RADIUS_MAX` the ≥2-oracle redundancy SPEC §3.1 tracks.

**It is exact, not a band.** On a disk the centre attains the maximum, so the value is the centre's
distance to the *nearest* boundary pixel — which is not the axial `(R, 0)` but the one at offset
`(1, R−1)`:

| | |
|---|---|
| it is inside | `1 + (R−1)² ≤ R²` ⟺ `2 ≤ 2R`, for every R ≥ 1 |
| it is boundary | its neighbour `(1, R)` has `1 + R² > R²`, so that one is background |
| it is nearer than `(R, 0)` | `(R−1)² + 1 = R² − 2R + 2 < R²` for every R > 1 |

so **`ROI_RADIUS_MAX = √((R−1)² + 1)`**, matched to 1e-12 relative. The `(1, R−1)` pixel was
confirmed nearest at R = 5, 10, 20, 40, 80 and 160, so the form is not fitted to the three radii the
test uses.

| R | ROI_RADIUS_MAX | √((R−1)²+1) | MAX/(R−1) |
|---|---|---|---|
| 10 | 9.0553851381374173 | 9.055385138137417 | 1.006154 |
| 20 | 19.026297590440446 | 19.026297590440446 | 1.001384 |
| 40 | 39.012818406262319 | 39.01281840626232 | 1.000329 |

The same form derives the convergence rather than fitting it: `MAX/(R−1) = √(1 + 1/(R−1)²) → 1` as
`1/(2(R−1)²)`, so MAX grows **linearly** in R.

## What the assertions discriminate

An oracle assertion is only worth its tolerance if something plausible fails it. Measured against an
implementation that returns the squared distance instead of the distance:

| | pinned | squared |
|---|---|---|
| ROI_RADIUS_MAX, R=10 | 9.055 | **82** |
| ROI_RADIUS_MEDIAN, R=10 | 2.236 | **5** |
| MAX/(R−1) at R = 10, 20, 40 | 1.006, 1.001, 1.000 — converging | **9.1, 19.1, 39.0 — growing** |

Both tests fail on all of it. The growth row is why three radii are pinned rather than one: on a
single fixture 9.055 and 82 are each just a number, and only the behaviour across R identifies which
one is a length.

## ROI_RADIUS_MEAN carries no oracle row, and the reason is measured

`ContourFeature::buildRegularContour` traces in an image padded by one pixel on every side and
restores only the bounding-box origin, never the pad, so every contour pixel is reported one pixel
right and one pixel down of where it is. `RoiRadiusFeature` therefore measures its pixels against a
**shifted** boundary. This is a live defect in `contour.cpp`, not in this family, and it reaches
`radial_distribution.cpp`, `circle.cpp`, the 62 weighted moments in `2d_geomoments_basic.cpp` and the
GPU rebasing in `cache.cpp` as well.

Shifting the reference boundary by (+1, +1) reproduces every Nyxus value exactly, on the disks and on
the 8×8 raster alike, which is what identifies the offset as the whole of the difference. The
coordinates say the same thing directly: on the 8×8 mask the traced Nyxus contour **is** the skimage
inner boundary shifted by (+1, +1), all 18 pixels, compared as sets rather than counts.

| fixture | MEAN | MEAN, shifted | MAX | MAX, shifted | MEDIAN | MEDIAN, shifted |
|---|---|---|---|---|---|---|
| disk R=10 | 2.7517095857010947 | **2.8334114624727413** | 9.055385138137417 | 9.055385138137417 | 2.23606797749979 | 2.23606797749979 |
| disk R=20 | 6.047026418093844 | **6.087109772358641** | 19.026297590440446 | 19.026297590440446 | 5.0 | 5.0 |
| disk R=40 | 12.686020245442752 | **12.705792452220727** | 39.01281840626232 | 39.01281840626232 | 11.045361017187261 | 11.045361017187261 |
| shape2d | 0.3076923076923077 | **0.6247169495045879** | 1.0 | 1.4142135623730951 | 0.0 | 1.0 |

The shifted column is what Nyxus reports. On a disk MAX and MEDIAN are **unmoved** by the shift — the
pixel attaining them moves with it — which is why those two carry oracle rows and MEAN does not. On
the 8×8 raster nothing survives the shift, which is why the shape2d rows are drift guards and the
vetting claim lives on the disks.

The generator prints that comparison on every run, so it is a standing measurement rather than a
note, and correcting the contour offset makes MEAN vettable against this same reference with a
registry edit and a re-pin.

## Registry

| feature | rows |
|---|---|
| ROI_RADIUS_MEAN | `regression` on `morphology.shape2d_native`, flag `blocked-by-contour-offset` |
| ROI_RADIUS_MAX | `vetted`/`skimage` **and** `vetted`/`analytic` on `morphology.radius_disks`; `regression` drift guard on `morphology.shape2d_native` |
| ROI_RADIUS_MEDIAN | `vetted`/`skimage` on `morphology.radius_disks`; `regression` drift guard on `morphology.shape2d_native` |

## Outside this round

`docs/source/Math/f_morphology.rst` defines these three features as statistics of the distance from
the **centroid** to each **edge pixel**, and `docs/source/featurelist.rst` repeats it as "centroid to
edge distance". The implementation measures a different quantity — for each ROI pixel, the distance
to the nearest contour pixel. On the disks above the documented quantity gives MEAN 9.57 / 19.54 /
39.53 where the implementation gives 2.75 / 6.05 / 12.69. Which of the two the feature is meant to be
is a question for the maintainers, not something this round settles.
