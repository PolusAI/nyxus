# 2D ROI_RADIUS vs scikit-image — vetting report

Fixes the defect that demoted `ROI_RADIUS_MEDIAN` in
`morphology_2d_skimage_vetting_report.md` — the three `ROI_RADIUS_*` features reported **squared**
distances — fixes a second one underneath it, and promotes two of the three rows.

## Tool and configuration

| | |
|---|---|
| Tool | scikit-image 0.26.0, scipy 1.17.1, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_morphology_radius_skimage.py` |
| Recipe | `morphology.radius_disks` |
| Benchmark | `bench_radius_disks` — filled digital disks at R = 10, 20, 40 |
| Tests | `test_2d_morphology_skimage.h` (`TEST_2D_MORPHOLOGY_ROI_RADIUS_DISKS_SKIMAGE`) and `test_2d_morphology_analytic.h` (`TEST_2D_MORPHOLOGY_ROI_RADIUS_DISK_CLOSED_FORM_ANALYTIC`) |
| Tolerance | `rel=1e-3` |

```
python tests/vetting/oracles/gen_morphology_radius_skimage.py
```

Verifies every golden pinned in the disk table of the test header and exits non-zero on mismatch, on
an empty table as well as on a wrong value. Current run: **6 verified, 0 failed, 0 unproducible**.

## The two defects

`RoiRadiusFeature::calculate` (`src/nyx/features/roi_radius.cpp`) fed `Pixel2::min_sqdist()` straight
into the mean, max and median. That function returns a **squared** distance, so all three features
reported squared distances under a name that says radius. `osized_calculate` had the same bug through
`min_max_sqdist`, whose minimum is the same call.

Underneath it, `min_sqdist()` is an approximate hill-descent whose own header comment says it "can
settle in the wrong basin on a closed contour, OVERESTIMATING the true minimum", and names the radius
gate as an affected caller. `exact_min_sqdist()` — a full linear scan — is defined directly beside it
and was already used by the neighbor features.

`ROI_RADIUS_MAX` on the 8×8 `shape2d_morphology` ROI shows both at once:

| | value |
|---|---|
| before | **4** |
| after removing the square | 2 |
| after also taking the exact minimum | **1.4142135623730951** |

4 is 2 squared, and 2 is the approximation overshooting an exact √2. Both paths now take
`sqrt(exact_min_sqdist(...))`, and the median is taken over the distances themselves rather than
through `TrivialHistogram`, whose item type is `unsigned int` and would have rounded every radius to
a whole pixel.

## The reference

`RoiRadiusFeature` measures every ROI pixel against the ROI's own contour and reports the mean, max
and median of those distances. The reference computes the same quantity from two library calls:

```
boundary = skimage.segmentation.find_boundaries(mask, connectivity=1, mode='inner')
radii    = min distance from each mask pixel to a boundary pixel
```

`find_boundaries(connectivity=1, mode='inner')` is the whole of the convention — the foreground
pixels 4-adjacent to background. It is **CellProfiler's own edge definition**, which is why
`gen_morphology_cellprofiler.py` and `gen_morphology_wholeslide_cellprofiler.py` already reproduce the
`EDGE_*` statistics through it. On the 8×8 mask it does not merely agree in *count* with the Nyxus
chain-code contour — measured pixel by pixel, the Nyxus contour **is** this boundary shifted by
(+1, +1), all 18 coordinates, exactly. On the r=20 disk it returns **112** boundary pixels, the count
`benchmarks.md` records for `bench_disk64_diagonal_boundary`, which is the same 1257-pixel shape.

The distance step carries no convention and is computed twice — an exhaustive minimum and
`scipy.ndimage.distance_transform_edt` over the complement of the boundary — with the two required to
agree, so neither is taken on trust.

**Disks and not the 8×8 raster**, for the reason `morphology.perimeter_circles` exists: on 26 pixels
with a hole the two boundary conventions have nothing to converge to. **Three radii and not one**,
because the defect is a units error and a single disk cannot see units — 82 and 9.055 are each just
"a number" on one disk. Across a doubling of R the reference grows 2.10× and 2.05×; a squared
distance grows 4.41× and 4.20×.

## Results

| R | pixels | feature | Nyxus | scikit-image | rel |
|---|---|---|---|---|---|
| 10 | 317 | ROI_RADIUS_MAX | 9.0553851381374173 | 9.055385138137417 | 0 |
| 10 | 317 | ROI_RADIUS_MEDIAN | 2.2360679774997898 | 2.23606797749979 | 0 |
| 20 | 1257 | ROI_RADIUS_MAX | 19.026297590440446 | 19.026297590440446 | 0 |
| 20 | 1257 | ROI_RADIUS_MEDIAN | 5.0 | 5.0 | 0 |
| 40 | 5025 | ROI_RADIUS_MAX | 39.012818406262319 | 39.01281840626232 | 0 |
| 40 | 5025 | ROI_RADIUS_MEDIAN | 11.045361017187261 | 11.045361017187261 | 0 |

## A second oracle, for what the first one cannot see

The pins say Nyxus agrees with a reference implementation of the definition. They do not say the
value obeys the geometry — a reference that shared a mistake would agree with it. So the closed form
is asserted separately, as its own `analytic` row on the same recipe
(`test_2d_morphology_analytic.h`, `TEST_2D_MORPHOLOGY_ROI_RADIUS_DISK_CLOSED_FORM_ANALYTIC`), which
also gives `ROI_RADIUS_MAX` the ≥2-oracle redundancy SPEC §3.1 tracks.

**The form is exact, not a band.** `ROI_RADIUS_MAX` is the largest distance from any ROI pixel to
the boundary, and on a disk the centre attains it, so the value is the centre's distance to the
*nearest* boundary pixel. That pixel is **not** the axial `(R, 0)` at distance R — it is the one at
offset `(1, R−1)`:

| | |
|---|---|
| it is inside | `1 + (R−1)² ≤ R²` ⟺ `2 ≤ 2R`, for every R ≥ 1 |
| it is boundary | its neighbour `(1, R)` has `1 + R² > R²`, so that one is background |
| it is nearer than `(R, 0)` | `(R−1)² + 1 = R² − 2R + 2 < R²` for every R > 1 |

so **`ROI_RADIUS_MAX = √((R−1)² + 1)`**, and Nyxus matches it to 1e-12 relative. The `(1, R−1)` pixel
was confirmed to be the nearest at R = 5, 10, 20, 40, 80 and 160, so the form is not fitted to the
three radii the test uses.

| R | ROI_RADIUS_MAX | √((R−1)²+1) | MAX/(R−1) | before the fix | ratio before |
|---|---|---|---|---|---|
| 10 | 9.0553851381374173 | 9.055385138137417 | 1.006154 | 82 | 9.11 |
| 20 | 19.026297590440446 | 19.026297590440446 | 1.001384 | 362 | 19.05 |
| 40 | 39.012818406262319 | 39.01281840626232 | 1.000329 | 1522 | 39.03 |

The same form explains the convergence: `MAX/(R−1) = √(1 + 1/(R−1)²) → 1` as `1/(2(R−1)²)`, i.e. MAX
grows **linearly** in R. A squared distance cannot: its ratio to R−1 is 9.1, 19.1, 39.0 — growing,
not converging. That is the whole content of "these are not radii", and an exact assertion at each
radius closes it more tightly than any band could.

## ROI_RADIUS_MEAN is not promoted, and the residual is measured

MEAN is the one of the three the reference does not reproduce, and the gap is not this family's.
`ContourFeature::buildRegularContour` traces in an image padded by one pixel on every side and
restores only the bounding-box origin, never the pad, so every contour pixel is reported one pixel
right and one pixel down of where it is. Nyxus therefore measures its pixels against a **shifted**
boundary.

Shifting the reference boundary by (+1, +1) reproduces every Nyxus value exactly, on the disks and on
the 8×8 raster alike — which is what identifies the offset as the whole of the difference. The
coordinates say the same thing directly: on the 8×8 mask the traced Nyxus contour **is** the skimage
inner boundary shifted by (+1, +1), all 18 pixels, compared as sets rather than counts.

| fixture | MEAN | MEAN, shifted | MAX | MAX, shifted | MEDIAN | MEDIAN, shifted |
|---|---|---|---|---|---|---|
| disk R=10 | 2.7517095857010947 | **2.8334114624727413** | 9.055385138137417 | 9.055385138137417 | 2.23606797749979 | 2.23606797749979 |
| disk R=20 | 6.047026418093844 | **6.087109772358641** | 19.026297590440446 | 19.026297590440446 | 5.0 | 5.0 |
| disk R=40 | 12.686020245442752 | **12.705792452220727** | 39.01281840626232 | 39.01281840626232 | 11.045361017187261 | 11.045361017187261 |
| shape2d | 0.3076923076923077 | **0.6247169495045879** | 1.0 | 1.4142135623730951 | 0.0 | 1.0 |

The shifted column is what Nyxus reports today. On a disk MAX and MEDIAN are **unmoved** by the
shift — the pixel attaining them simply moves with it — which is why those two promote now and MEAN
cannot. On the 8×8 raster nothing survives the shift, which is why the shape2d rows stay drift guards
and the vetting claim lives on the disks.

The generator prints this table on every run, so it is a standing measurement rather than a note.
Fixing the contour offset is its own change: it reaches `radial_distribution.cpp`, `circle.cpp`, the
62 weighted moments in `2d_geomoments_basic.cpp` and the GPU rebasing in `cache.cpp`, none of which
are this family's.

## Registry

| feature | before | after |
|---|---|---|
| ROI_RADIUS_MEAN | `regression`, agreement `needs_audit` | `regression`, agreement `disagrees`, flag `blocked-by-contour-offset` |
| ROI_RADIUS_MAX | `regression`, agreement `needs_audit` | **`vetted` / `skimage`** *and* **`vetted` / `analytic`** on `morphology.radius_disks`, plus its shape2d drift guard |
| ROI_RADIUS_MEDIAN | `regression`, agreement `disagreed` | **`vetted` / `skimage`** on `morphology.radius_disks`, plus its shape2d drift guard |

The three shape2d goldens moved with the fix and were re-pinned: MEAN 1.07692307692308 →
0.62471694950458789, MAX 4 → 1.4142135623730951, MEDIAN 1 → 1 (√1 = 1 is a coincidence of this
fixture, not a value that did not move).

## Negative control

The new oracle test was run against the pre-fix implementation. It fails on all three counts:
`ROI_RADIUS_MAX` comes back as **82** against a pinned 9.055, `ROI_RADIUS_MEDIAN` as 5 against
2.236, and the closed-form assertion reports `MAX/(R-1)` off by 8.11 at R = 10.

## Not addressed here

`docs/source/Math/f_morphology.rst` defines these three features as statistics of the distance from
the **centroid** to each **edge pixel**, and `docs/source/featurelist.rst` repeats it as "centroid to
edge distance". The code computes something else — for each ROI pixel, the distance to the nearest
contour pixel — and did so before this change as well. That is a documentation-versus-code question
about what the feature is meant to be, not a units bug, so it is left to be answered rather than
silently decided here.
