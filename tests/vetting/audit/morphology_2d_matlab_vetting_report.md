# 2D morphology MATLAB R2026a vetting report

This replaces the GNU Octave surrogate provenance for the 33 existing `oracle=matlab` assertions
with a licensed MATLAB run. The assertion set is unchanged: MATLAB R2026a reproduces every pinned
feature, with a maximum relative change of `3.40e-16` from the former Octave-generated literals.

## Provenance

| | |
|---|---|
| Tool | MATLAB R2026a, Image Processing Toolbox 26.1 |
| Generator | `tests/vetting/oracles/gen_morphology_matlab.m` |
| Fixture | `shape2d_morphology_{mask,intensity}` from `tests/test_data.h` on `PolusAI/nyxus` main |
| Recipe | `morphology.shape2d_native` |
| Nyxus config | `make_shape2d_settings()` — `PIXELSIZEUM=2.0`, `XYRES=1.0`, `GREYDEPTH=128`, `IBSI=false`, single ROI |
| Test | `test_2d_morphology_matlab.h` |
| Tolerance | `rel=1e-3` (SPEC §7 same-definition oracle tier) |

Run from the repository root:

```text
matlab -batch "run('tests/vetting/oracles/gen_morphology_matlab.m')"
```

The generator downloads the fixture and the two C++ golden tables from the moving
`PolusAI/nyxus` `main` tree, evaluates the named MATLAB built-ins, and checks the feature names in
both directions. A pin that MATLAB does not produce, or a MATLAB value with no corresponding pin,
is an error. The verified R2026a run produced **33 values**, with none missing and none outside
`rel=1e-3`.

## What MATLAB computes

| Nyxus features | MATLAB source |
|---|---|
| `AREA_PIXELS_COUNT` | `regionprops(..., 'Area')` |
| `CENTROID_X/Y` | `regionprops(..., 'Centroid')` |
| `WEIGHTED_CENTROID_X/Y` | `regionprops(mask, intensity, 'WeightedCentroid')` |
| `BBOX_XMIN/YMIN`, `BBOX_WIDTH/HEIGHT` | `regionprops(..., 'BoundingBox')` |
| `EXTENT` | `regionprops(..., 'Extent')` |
| `MAJOR_AXIS_LENGTH`, `MINOR_AXIS_LENGTH`, `ECCENTRICITY` | matching `regionprops` properties |
| `EULER_NUMBER` | `bweuler(mask, 8)` |
| `EXTREMA_P1..P8_X/Y` | `regionprops(..., 'Extrema')` |

Three derived Nyxus definitions use only direct arithmetic over those built-in results:

- `AREA_UM2 = Area * PIXELSIZEUM^2`
- `ASPECT_RATIO = BoundingBox width / height`
- `ELONGATION = MinorAxisLength / MajorAxisLength`

No Nyxus feature algorithm is reproduced in the generator.

## Coordinate conventions

- MATLAB centroids are 1-based pixel centres; Nyxus uses 0-based pixel centres, so the recipe
  subtracts 1.
- MATLAB `BoundingBox` begins at the upper-left sub-pixel corner. Subtracting 0.5 from its 1-based
  origin gives the Nyxus minimum pixel index; width and height are unchanged.
- MATLAB `Extrema` returns eight 1-based sub-pixel corners in its documented fixed order. Mapping
  them to 0-based pixel centres subtracts 0.5 for a left/top edge and 1.5 for a right/bottom edge.

These are coordinate-frame conversions, not tuned offsets.

## Measured agreement

Twenty-eight values are bit-for-bit identical to the former pins. The remaining five differ only
at the final floating-point digit:

| Feature | MATLAB R2026a | former pin | relative difference |
|---|---:|---:|---:|
| `CENTROID_X` | 2.6153846153846154 | 2.6153846153846163 | 3.40e-16 |
| `CENTROID_Y` | 2.8461538461538463 | 2.8461538461538467 | 1.56e-16 |
| `MINOR_AXIS_LENGTH` | 5.48870991295738 | 5.4887099129573791 | 1.62e-16 |
| `ELONGATION` | 0.78761008754746209 | 0.78761008754746198 | 1.41e-16 |
| `ECCENTRICITY` | 0.61617396082070774 | 0.61617396082070786 | 1.80e-16 |

All are many orders of magnitude inside `rel=1e-3`.

## Why these ellipse features use MATLAB

MATLAB and Nyxus apply the same `+1/12` finite-pixel correction to the normalized second central
moments. Therefore `MAJOR_AXIS_LENGTH`, `MINOR_AXIS_LENGTH`, and `ECCENTRICITY` have matching
definitions here; scikit-image omits that correction and differs by about 1.4% on this fixture.
