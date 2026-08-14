# 2D morphology vs MATLAB (GNU Octave) — vetting report

Closes the 17 `oracle=matlab` rows that read `status=vetted` with no in-tree oracle assertion, plus
the 16 `EXTREMA_*` rows that were already asserted here but never re-derived from a fresh tool run.

## Tool and configuration

| | |
|---|---|
| Tool | GNU Octave 11.3.0 + `image` package 2.20.0, as the license-free MATLAB stand-in (TOOLS.md) |
| Generator | `tests/vetting/oracles/gen_morphology_matlab.m` |
| Recipe | `morphology.shape2d_native` |
| Fixture | `shape2d_morphology_{mask,intensity}`, read out of `tests/test_data.h` by the generator |
| Nyxus config | `make_shape2d_settings()` — `PIXELSIZEUM=2.0`, `XYRES=1.0`, `GREYDEPTH=128`, `IBSI=false`, single ROI |
| Test | `test_2d_morphology_matlab.h` |
| Tolerance | `rel=1e-3` (SPEC §7 same-definition oracle) |

Reproduce, from the repository root:

```
octave tests/vetting/oracles/gen_morphology_matlab.m
```

The generator prints the paste-ready table, then re-verifies **every** golden pinned in
`test_2d_morphology_matlab.h` — both tables — and exits non-zero on any mismatch or on any pin it
cannot produce. Current run: **33 verified, 0 failed, 0 unproducible**, every one at `rel = 0`.

## Results

| feature | Nyxus | Octave `regionprops` | rel |
|---|---|---|---|
| AREA_PIXELS_COUNT | 26 | 26 | 0 |
| AREA_UM2 | 104 | 104 | 0 |
| CENTROID_X | 2.6153846153846163 | 2.6153846153846163 | 0 |
| CENTROID_Y | 2.8461538461538467 | 2.8461538461538467 | 0 |
| WEIGHTED_CENTROID_X | 2.8416030534351147 | 2.8416030534351147 | 0 |
| WEIGHTED_CENTROID_Y | 3.4389312977099236 | 3.4389312977099236 | 0 |
| BBOX_XMIN / BBOX_YMIN | 0 / 0 | 0 / 0 | 0 |
| BBOX_WIDTH / BBOX_HEIGHT | 6 / 7 | 6 / 7 | 0 |
| ASPECT_RATIO | 0.8571428571428571 | 0.8571428571428571 | 0 |
| EXTENT | 0.61904761904761907 | 0.61904761904761907 | 0 |
| MAJOR_AXIS_LENGTH | 6.9688161689861872 | 6.9688161689861872 | 0 |
| MINOR_AXIS_LENGTH | 5.4887099129573791 | 5.4887099129573791 | 0 |
| ELONGATION | 0.78761008754746198 | 0.78761008754746198 | 0 |
| ECCENTRICITY | 0.61617396082070786 | 0.61617396082070786 | 0 |
| EULER_NUMBER | 0 | 0 (`bweuler` at both 4- and 8-connectivity) | 0 |
| EXTREMA_P1..P8 _X/_Y (16) | see the test header | `regionprops('Extrema')` + the corner offset below | 0 |

## Conventions applied, and why

These are part of the recipe, not fudge factors — each is a documented frame difference.

- **Centroid / WeightedCentroid.** MATLAB reports 1-based pixel centres, Nyxus 0-based: subtract 1.
- **BoundingBox.** MATLAB returns `[x_ul y_ul w h]` with the corner at `min - 0.5` in 1-based
  coordinates, so the 0-based minimum index is `BoundingBox(1) - 0.5`. The widths need no change.
- **AREA_UM2** is `Area * PIXELSIZEUM^2` = 26 × 4.
- **ASPECT_RATIO** and **ELONGATION** are ratios of two `regionprops` outputs (bbox w/h, and
  minor/major), not properties of their own.
- **Extrema.** `regionprops('Extrema')` returns 8 sub-pixel *corner* points, 1-based, ordered
  top-left, top-right, right-top, right-bottom, bottom-right, bottom-left, left-bottom, left-top.
  Nyxus returns 0-based pixel *centres*, so the offset is direction-specific: a left or top
  coordinate maps as `matlab - 0.5`, a right or bottom one as `matlab - 1.5`. That rule was
  previously asserted only in a comment; the generator now derives all 16 values from a live
  `regionprops` call and reproduces the pinned goldens exactly.

## Why the ellipse triple vets here and not against scikit-image

`MAJOR_AXIS_LENGTH`, `MINOR_AXIS_LENGTH` and `ECCENTRICITY` match MATLAB to ~1e-15 but differ from
scikit-image by ~1.4% (`axis_major` 6.872 vs 6.969; `eccentricity` 0.625 vs 0.616). The cause is the
pixel finite-size correction: MATLAB and Nyxus both add 1/12 to the normalised second central
moments to account for a pixel being a unit square rather than a point; scikit-image does not.

This is why the registry pins these three to `oracle=matlab`. `ORIENTATION` is the exception that
proves the rule — the correction shifts `mu20` and `mu02` equally, leaving `mu20-mu02` and `mu11`
unchanged, so the *angle* is invariant to it and vets against scikit-image instead. Octave
independently confirms that value: it returns `-70.417394498420691` against the Nyxus
`+70.4173944984207`, the same magnitude to 13 digits with the sign carrying the y-axis-direction
convention.

## What did not hold up

`test_2d_morphology_perimeter_matlab()` asserted `PERIMETER = 999.26` against a documented recipe of
`nnz(bwperim(imfill(circles.png)))`. Run fresh, that recipe returns **846** — it counts perimeter
*pixels* — and MATLAB's actual `regionprops('Perimeter')` returns **952.848**. Neither is the pinned
number. The golden turned out to be scikit-image's `measure.perimeter` (999.259018078045, agreeing
with Nyxus to 3.8e-15), so the assertion moved to `test_2d_morphology_skimage.h` and the registry row
now names the oracle that actually backs it. See `morphology_2d_skimage_vetting_report.md`.
