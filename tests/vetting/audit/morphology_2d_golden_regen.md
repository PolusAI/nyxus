# Regenerating the 2D morphology oracle goldens

Every generator here is offline; CI never invokes one. Each prints a paste-ready table **and**
re-verifies every golden currently pinned in the header it feeds, exiting non-zero on any mismatch
or on any pin it cannot produce. Run them from the repository root.

## Environments

| generator | interpreter | how to get it |
|---|---|---|
| `gen_morphology_matlab.m` | GNU Octave 11.3.0 + `image` 2.20.0 | the license-free MATLAB stand-in; see TOOLS.md for the launcher caveats |
| `gen_morphology_skimage.py` | conda env `nyxus_mirp` (scikit-image 0.26.0, numpy) | `conda create -n nyxus_mirp -c conda-forge python=3.12 scikit-image numpy opencv` |
| `gen_morphology_imea.py` | conda env `nyxus_mirp` + `pip install imea==0.3.5` | imea pulls opencv and pandas |

Fixtures are **not** copied into the generators. All three parse `tests/test_data.h` for the
`{x, y, value}` pixel arrays, so the generator and the C++ tests share one copy of the pixels — the
same discipline as `oracles/ibsi_phantom.py`. The ellipse benchmark is the one exception: it is
generated arithmetically from `a=20, b=10, cx=26, cy=16`, mirroring
`calculate_ellipse_caliper_values()` in `test_2d_morphology_common.h`.

## MATLAB / Octave — `test_2d_morphology_matlab.h`

```
octave tests/vetting/oracles/gen_morphology_matlab.m
```

Expect `33 verified, 0 failed, 0 unproducible`.

Mapping `regionprops` output to Nyxus names, all of it part of recipe `morphology.shape2d_native`:

| Nyxus | from | conversion |
|---|---|---|
| `AREA_PIXELS_COUNT` | `Area` | — |
| `AREA_UM2` | `Area` | × `PIXELSIZEUM²` (= 4) |
| `CENTROID_X/Y` | `Centroid` | − 1 (1-based centres → 0-based) |
| `WEIGHTED_CENTROID_X/Y` | `WeightedCentroid` (intensity image) | − 1 |
| `BBOX_XMIN/YMIN` | `BoundingBox(1:2)` | − 0.5 (corner at min−0.5, 1-based) |
| `BBOX_WIDTH/HEIGHT` | `BoundingBox(3:4)` | — |
| `ASPECT_RATIO` | `BoundingBox(3)/BoundingBox(4)` | — |
| `EXTENT` | `Extent` | — |
| `MAJOR/MINOR_AXIS_LENGTH` | same names | — |
| `ELONGATION` | `MinorAxisLength/MajorAxisLength` | — |
| `ECCENTRICITY` | `Eccentricity` | — |
| `EULER_NUMBER` | `bweuler(M, 8)` | — |
| `EXTREMA_P1..P8_X/Y` | `Extrema` | per-point corner offset (below) |

**Extrema offsets.** `Extrema` returns 8 sub-pixel corners, 1-based, ordered top-left, top-right,
right-top, right-bottom, bottom-right, bottom-left, left-bottom, left-top. Nyxus returns 0-based
pixel centres, so a *left or top* coordinate maps as `matlab − 0.5` and a *right or bottom* one as
`matlab − 1.5`. In generator terms: `dx = [-0.5 -1.5 -1.5 -1.5 -1.5 -0.5 -0.5 -0.5]`,
`dy = [-0.5 -0.5 -0.5 -1.5 -1.5 -1.5 -1.5 -0.5]`. A uniform −0.5 is wrong and was a past harness bug.

**Convention to know.** MATLAB and Nyxus both apply the +1/12 pixel finite-size correction to the
normalised second central moments; scikit-image does not. That is why the axis lengths and
eccentricity are vetted here and not against skimage, and why `ORIENTATION` — invariant to the
correction — is the reverse.

## scikit-image — `test_2d_morphology_skimage.h`

```
python tests/vetting/oracles/gen_morphology_skimage.py
```

Expect `6 verified, 0 failed, 0 unproducible`.

| Nyxus | from | note |
|---|---|---|
| `CONVEX_HULL_AREA` | `convex_hull_image(mask, offset_coordinates=False).sum()` | Nyxus hulls through pixel centres; the regionprops default (`offset_coordinates=True`) rasterises to 28, not 27 |
| `SOLIDITY` | area / hull area | — |
| `ORIENTATION` | `regionprops.orientation` | `90 − degrees(...)`: skimage measures from the row axis, Nyxus from x |
| `EROSIONS_2_VANISH` | `erosion(m, footprint_rectangle((3,3)))` until empty | 8-connected; `disk(1)` gives 2, not 1 |
| `DIAMETER_EQUAL_AREA` | `regionprops.equivalent_diameter_area` | `sqrt(4A/π)` |
| `PERIMETER` | `measure.perimeter(mask)` | **circles fixture only** — recipe `morphology.perimeter_circles` |

`PERIMETER` is not comparable on the small shape2d mask (26.935 vs 12.657). Do not regenerate it
there. Note also that `nnz(bwperim(...))` counts perimeter *pixels* (846) and MATLAB's
`regionprops('Perimeter')` returns 952.848 — neither is this quantity.

## imea — `test_2d_morphology_imea.h`

```
python tests/vetting/oracles/gen_morphology_imea.py
```

Expect `20 verified, 0 failed, 0 unproducible`, plus two evidence blocks printed for the record.

Two benchmarks with different scopes:

- **shape2d, the ISO transforms only.** `macro.perimeter_equal_diameter(P)` and
  `macro.geodeticlength_and_thickness(A, P)` fed the **Nyxus** area and perimeter. This vets the
  transform, not its inputs — imea's end-to-end values use `cv2.arcLength` (12.657 vs the Nyxus
  26.935) and do not agree.
- **the a=20/b=10 ellipse, for the calipers.**
  `imea.shape_measurements_2d(mask, spatial_resolution_xy=1.0, dalpha=10)`.

**Use `dalpha=10`.** It matches Nyxus' own `rot_angle_increment` (`src/nyx/features/caliper.h`), so
both sample the same angles. Mixing steps is what left the previous goldens inconsistent and forced
an inflated tolerance.

Name mapping: `martin_{min,max,mean,median,std}`, `nassenstein_*`, `feret_*` →
`STAT_{MARTIN,NASSENSTEIN,FERET}_DIAM_{MIN,MAX,MEAN,MEDIAN,STDDEV}`; `allchords_min` →
`ALLCHORDS_MIN`; `diameter_min_enclosing_circle` → `DIAMETER_MIN_ENCLOSING_CIRCLE`.

**Do not regenerate as oracle values:**

- the `_MODE` statistics — imea's own mode ranges over 19..24 as `dalpha` goes 5 → 30, further than
  the Nyxus-imea gap. They are regression rows.
- any caliper statistic on the 8×8 shape2d raster — imea's values there differ from Nyxus by
  3.9–79.3%. Those are regression snapshots in `test_2d_morphology_regression.h`.

## Coverage artifact

`morphology_2d_coverage.csv` is generated, not hand-written:

```
python tests/vetting/audit/scan_morphology_coverage.py           # rewrite
python tests/vetting/audit/scan_morphology_coverage.py --check   # CI-style drift + acceptance check
```

`--check` also enforces the family acceptance rule: no `vetted` row without an oracle assertion, no
row naming an oracle nothing compares against, and `current_test` naming exactly the files that
cover the feature. Note its deliberate limit — it attributes an oracle from the function-name
suffix, so it cannot tell an oracle test asserting real tool output from one asserting a snapshot.
Only running the generator does that.
