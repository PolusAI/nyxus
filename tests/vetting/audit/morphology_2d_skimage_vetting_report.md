# 2D morphology vs scikit-image — vetting report

Closes the two `oracle=skimage` gap rows (`DIAMETER_EQUAL_AREA`, `ROI_RADIUS_MEDIAN`), re-homes
`PERIMETER` to the oracle that actually backs it, and puts the previously inline-pinned
`ORIENTATION` / `EROSIONS_2_VANISH` goldens under generator verification.

## Tool and configuration

| | |
|---|---|
| Tool | scikit-image 0.26.0, numpy 2.4.6, env `nyxus_mirp` (conda) |
| Generator | `tests/vetting/oracles/gen_morphology_skimage.py` |
| Recipes | `morphology.shape2d_native`, `morphology.perimeter_circles` |
| Fixtures | `shape2d_morphology_mask` and `roiDataForPerimeterTest`, read out of `tests/test_data.h` |
| Test | `test_2d_morphology_skimage.h` |
| Tolerance | `rel=1e-3`, except the hull pair at `rel=1e-2` |

```
python tests/vetting/oracles/gen_morphology_skimage.py
```

Verifies every golden pinned in both tables of the test header and exits non-zero on mismatch.
Current run: **6 verified, 0 failed, 0 unproducible**.

## Results

| feature | Nyxus | scikit-image | rel | source |
|---|---|---|---|---|
| DIAMETER_EQUAL_AREA | 5.7536273917515919 | 5.753627391751592 | 0 | `regionprops.equivalent_diameter_area` |
| ORIENTATION | 70.417394498420663 | 70.417394498420663 | 0 | `90 - degrees(regionprops.orientation)` |
| EROSIONS_2_VANISH | 1 | 1 | 0 | `erosion(footprint_rectangle((3,3)))` count |
| CONVEX_HULL_AREA | 27 | 27 | 0 | `convex_hull_image(offset_coordinates=False).sum()` |
| SOLIDITY | 0.9629629629629629 | 0.9629629629629629 | 0 | area / hull area |
| PERIMETER | 999.25901807804871 | 999.25901807804496 | 3.8e-15 | `measure.perimeter`, circles fixture |

`DIAMETER_EQUAL_AREA` is `sqrt(4·Area/π)` on both sides over an exact pixel count, so agreement is
exact rather than merely close.

## PERIMETER: right number, wrong oracle

This assertion previously lived in `test_2d_morphology_matlab.h` under the recipe
`nnz(bwperim(imfill(circles.png)))`, documented in the header as returning 846. Running all three
candidates on the fixture:

| quantity | value |
|---|---|
| Nyxus `PERIMETER` | 999.25901807804871 |
| skimage `measure.perimeter` / `regionprops.perimeter` | 999.25901807804496 |
| MATLAB `regionprops('Perimeter')` | 952.848 |
| `nnz(bwperim(M, 4))` | 846 |
| `nnz(bwperim(M, 8))` | 1216 |

The pinned 999.26 was a truncated copy of the scikit-image value, filed under a MATLAB recipe that
computes a different quantity entirely. Nyxus' chain-code contour walk and scikit-image's
4-neighbourhood boundary walk are the same algorithm, and on a 14309-pixel object they agree to
3.8e-15. The assertion now sits in the skimage file at full precision.

They do **not** agree on the small `shape2d` mask — 26.935 vs 12.657 — because that object is 26
pixels with a one-pixel hole and the two contour conventions have nothing to converge to. PERIMETER
is therefore vetted on the circles benchmark only and stays a regression row on shape2d, which is
what recipe `morphology.perimeter_circles` records.

## ROI_RADIUS_MEDIAN: demoted, the feature reports squared distances

The claim could not be reproduced, and the reason is a defect rather than a convention gap.

`RoiRadiusFeature::calculate` (`src/nyx/features/roi_radius.cpp:22-36`) feeds `Pixel2::min_sqdist()`
straight into the mean, max and median. That function returns a **squared** distance
(`src/nyx/features/pixel.cpp:13`), so all three `ROI_RADIUS_*` features report squared distances
under a name that says radius.

Measured on filled disks, where the pixel farthest from the boundary is the centre at distance R:

| R | ROI_RADIUS_MAX | (R−1)² | √MAX |
|---|---|---|---|
| 10 | 82 | 81 | 9.06 |
| 20 | 362 | 361 | 19.03 |
| 40 | 1522 | 1521 | 39.01 |

It scales as R², not R. No scikit-image radius statistic reproduces that, so `ROI_RADIUS_MEDIAN` is
demoted to `status=regression` with this measurement as the reason.

A second, milder defect sits underneath: `min_sqdist()` is an approximate hill-descent whose own
header comment states it "can settle in the wrong basin on a closed contour, OVERESTIMATING the true
minimum", and names the radius gate as an affected caller. `exact_min_sqdist()` is defined directly
beside it and is already used by the neighbor features.

Both are behaviour changes affecting three public feature values and their pins, so they are left to
a dedicated branch rather than folded into a vetting PR.
