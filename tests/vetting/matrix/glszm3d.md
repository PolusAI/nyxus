# 3D GLSZM config matrix

Axes = the settings `D3_GLSZM_feature::calculate()` actually reads; verdicts are measured, not
assigned (SPEC §5.1). There are exactly two, and they are coupled: `calculate()` reads
`STNGS_GLSZM_GREYDEPTH` and `STNGS_IBSI`, and overwrites `greyInfo` with `0` when `IBSI=true`
(`3d_glszm.cpp`). `GREYDEPTH` is read by neither, so the value a recipe records for it is inert here.

`GLSZM_GREYDEPTH` carries the binning scheme in its sign (`texture_feature.h`): `0` = no binning
(the IBSI reading of the raw levels), `> 0` = MATLAB binning at that level count, `< 0` = radiomics
binning at `abs()` bins.

**Two things that look like knobs and are not.** Zone connectivity is fixed at the full 26-voxel
neighbourhood, spelled out as 8 in-slice, 8 upper, 8 lower and 2 strictly-vertical offsets in
`gather_size_zones()` — it is not settable, so it is not a matrix axis, and the way to test it is a
fixture whose zones can be counted by hand (see below). Zone *size* likewise has no parameter: it is
the voxel count of a connected component, a property of the volume rather than a setting. 3D GLSZM
therefore has no distance, angle or alpha axis at all, unlike 3D GLCM, 3D GLRLM or 3D NGLDM.

| ibsi | binning | levels | verdict | recipe / oracle |
|---|---|---|---|---|
| False | radiomics (`GLSZM_GREYDEPTH < 0`) | 20 bins | VALID | `glszm3d.pyradiomics_bincount20` — pyradiomics, 16/16 |
| False | MATLAB (`GLSZM_GREYDEPTH > 0`) | 64 | VALID-prod-only | drift guard, `glszm3d.regression_ut_phantom` — no oracle |
| False | none (`GLSZM_GREYDEPTH == 0`) | the raw levels | VALID | `glszm3d.pyradiomics_ibsi_gapped` — pyradiomics, 16/16 + the matrix, on `bench_cube3_gapped_levels`; pinned on the phantom too, `glszm3d.regression_ut_phantom_nobinning` |
| True | none (forced) | the raw levels | VALID — the same cell | measured identical to the row above, values and matrix, by `test_3d_glszm_ibsi_equals_no_binning_mechanics`; see below |
| any | any | ROI with `aux_min == aux_max` | **DIVERGENCE** | a constant-intensity ROI has a valid one-zone GLSZM; `calculate()` returns the soft-NaN sentinel for all sixteen instead. Pinned, not endorsed, by `test_3d_glszm_constant_roi_regression`; see below |

## Measured agreement at the radiomics-binning cell

PyRadiomics 3.0.1 at `binCount=20`, no resampling, `weightingNorm=None`, against Nyxus at
`GLSZM_GREYDEPTH=-20`, `IBSI=false`, on `bench_compat_liver_3d`: fifteen of the sixteen features
agree to the last bit (worst relative residual 1.2e-15, `3GLSZM_GLV`), asserted at `rel=1e-9`.
`3GLSZM_ZE` sits at 1.9e-4 because `calc_sums_of_P()` takes the family's only logarithm through
`fast_log10(...)/LOG10_2`; it asserts at `rel=1e-3`, the documented-residual tier.

**The cell is measured on the matrix, not only on the sixteen scalars.** Every feature of this family
is a contraction of one `P(i, j)` size-zone matrix, so two errors inside it that cancel would leave
all sixteen scalars unmoved. `test_3d_glszm_matrix_pyradiomics` pins all 186 non-empty cells of the
20×634 matrix against PyRadiomics' own `P_glszm`, plus its `Ng`, `Ns`, `Nz`, `Np`, the width and
height it was allocated at, and a count that no cell outside those 186 is populated.

**And it is the production matrix.** The assertion takes `P` and `I` off the `D3_GLSZM_feature` the
sixteen scalar assertions ran, through the read-only accessors `3d_glszm.h` now carries, rather than
rebuilding a table beside `calculate()` from `gather_size_zones()`. That distinction is not
decorative: a defect in the row mapping, in `Ng`/`Ns`, in the allocation or in the fill loop lives
only in the object, and a rebuilt copy is blind to all four. Negative control: dropping one zone size
from the fill loop leaves `gather_size_zones()` correct and fails this assertion naming the level and
the size.

## The `IBSI=true` half is one cell, and it is the cell above it

`IBSI=true` makes `GLSZM_GREYDEPTH` inert — `calculate()` sets `greyInfo = 0` regardless of what was
passed — so that half of the cross-product collapses to a single point. It used to be recorded here
as NOT MEASURED, on the reasoning that it is a genuinely different branch: at `IBSI=true` the matrix
row index is `zone.first - 1` and `Ng` is `max_element(I)` rather than `I.size()`, and on a fixture
whose levels are contiguous from 1 the two coincide "by accident".

They do not coincide by accident, and the difference is not reachable at all. At `greyInfo == 0` the
`ibsi_grey_binning` branch builds `I` as the contiguous run `1..max(D)` whatever the volume holds —
absent levels get empty rows rather than being packed out — so the position of a level in `I` is
always `level - 1`, and `I.size()` is always `max(I)`. The two branches compute the same numbers on
every input, because `I` is constructed rather than gathered here.

That is now measured rather than argued:

- `glszm3d.pyradiomics_ibsi_gapped` runs the family at `IBSI=true` on `bench_cube3_gapped_levels`, a
  3×3×3 fixture carrying levels 1, 3 and 5 — the gapped case the old note said would be needed.
  All sixteen features reproduce PyRadiomics (`3GLSZM_ZE` at `rel=2e-3`, the `fast_log10` residual),
  and the size-zone matrix is pinned cell by cell off the feature object. PyRadiomics reports
  `Ng = 5` for three occupied levels; so does Nyxus.
- `test_3d_glszm_ibsi_equals_no_binning_mechanics` runs the same fixture at `IBSI=true,
  GLSZM_GREYDEPTH=64` and at `IBSI=false, GLSZM_GREYDEPTH=0` and asserts the sixteen values and the
  whole matrix are bit-identical. Passing 64 on the IBSI side also measures the overwrite: a run that
  honoured it would bin the levels into 64 MATLAB bins and miss every pin.

Negative control for both: removing the upper-Z offset block from `gather_size_zones()` fails the
gapped assertion and the connectivity one.

## `GLSZM_GREYDEPTH == 0` is a third configuration, not a variant of the other two

At MATLAB binning the background level is bin 1 (`zeroI = 1`), so a voxel binned there starts no zone
of its own; at radiomics binning and at no binning the background is intensity 0. That is a
different quantity, not a different resolution of the same one, which is why the recipes state the
value explicitly instead of leaning on the zero a settings vector starts at.

The unwired version of `test_3d_glszm_regression.h` did lean on it: it set `GREYDEPTH=64` and left
`GLSZM_GREYDEPTH` at 0, and at that setting every one of its sixteen pins is wrong by between 1.6×
and 1.5e6× (`tests/vetting/audit/glszm_3d_pyradiomics_vetting_report.md`, §"What the orphan would
have done").

**It is the default a real run gets, so it carries numbers and not only a finiteness check.**
`test_3d_glszm_default_greydepth_mechanics` asserts that `compile_feature_settings()` really does
deliver `GLSZM_GREYDEPTH = 0` and that the sixteen features are finite there — that is mechanics, and
mechanics is all it is. The values themselves are pinned by `glszm3d.regression_ut_phantom_nobinning`,
sixteen `%.17g` goldens on `bench_ut57_3d` under `TEST_3D_GLSZM_DEFAULT_GREYDEPTH_REGRESSION`. They
claim no oracle: PyRadiomics discretises before it counts zones and has no counterpart for reading
2001 raw levels off this phantom. The *setting* is vetted, on the fixture small enough to carry an
oracle — `glszm3d.pyradiomics_ibsi_gapped`, the row above.

## A constant-intensity ROI is not a blank one

`calculate()` opens with `if (r.aux_min == r.aux_max) { invalidate(...); return; }`, and the comment
above it reads "intercept blank ROIs (equal intensity)". Those are two different things. An ROI whose
voxels all carry one intensity is fully populated: at no binning or MATLAB binning it has one grey
level, one 26-connected zone if its voxels touch, a size-zone matrix with a single populated cell,
and sixteen finite features over it — on a 2×2×2 block, `SAE = 1/64`, `ZE = 0`, `GLV = 0`, `ZP = 1/8`.
Nyxus returns the soft-NaN sentinel for all sixteen instead, which is `--noval` and defaults to `0.0`.

**Recorded as a divergence, not as an invalid cell.** The guard is doing real work at radiomics
binning, where `to_grayscale_radiomix` divides by `(max - min)` and would divide by zero; it is
unconditional, so it also discards the two schemes that would have answered. Whether to narrow it is
`src/` work on its own branch — it changes a public feature's output on a reachable input — so this
pass pins the current behaviour rather than changing it: `test_3d_glszm_constant_roi_regression` runs
a constant 2×2×2 volume with the sentinel set to a distinctive `-98765` and asserts all sixteen come
back as that, which is an assertion about the intercept path rather than about a zero-filled buffer.
Filed in `PR/todo.md`.

## The connectivity check is a fixture, not a cell

The 26-neighbourhood is not selectable, so no row of this matrix exercises it as a config point. A
4×4×3 fixture does: `bench_cube4x4x3_zcross` (`glszm_3d_zcross_volume`), three populated slices whose
nine zones over four grey levels can be counted by eye and whose matrix
`test_3d_glszm_smallmatrix_pyradiomics` asserts cell by cell. The phantom cell above shows the same
property at scale, on 186 cells at once, but nobody can check those by hand — which is the division
of labour the two assertions exist for.

**The fixture separates the neighbourhoods it could be confused with, which its predecessor did not.**
That earlier version had one populated slice between two empty ones, so every `dz != 0` neighbour of
every voxel was background: 26-, 18- and 2D 8-connectivity all produced the same nine zones, and only
6-connectivity differed. It was measuring in-slice connectivity under a 3D name. The present one
carries a strictly vertical run, a z-edge join, a z-corner join no 18-neighbourhood makes, and two
same-level voxels two slices apart that must stay two zones; the four readings give 9, 10, 13 and 13
zones respectively. `bench_cube3_gapped_levels` carries the same properties in miniature.

Both volumes are fixtures rather than copies of one: there is no file to read them from, and
`gen_glszm3d_pyradiomics.py` runs PyRadiomics on the same literals at `binWidth=1`. That is what
keeps them out of SPEC §5.2's hand-labelled self-consistency — their expected values come from an
independent tool, not from the model that wrote them.
