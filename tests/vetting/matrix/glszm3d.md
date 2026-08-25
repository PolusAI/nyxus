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
| False | none (`GLSZM_GREYDEPTH == 0`) | the raw levels | REACHABLE, NOT VETTED | the default a run with no `--3glszm/greydepth` gets; asserted finite only, in `test_3d_glszm_mechanics.h` |
| True | none (forced) | the raw levels | NOT MEASURED | see below |
| any | any | ROI with `aux_min == aux_max` | INVALID | degenerate — a blank ROI has no zones; `calculate()` returns NaN before building a matrix, guarded in code rather than by an oracle |

## Measured agreement at the VALID point

PyRadiomics 3.0.1 at `binCount=20`, no resampling, `weightingNorm=None`, against Nyxus at
`GLSZM_GREYDEPTH=-20`, `IBSI=false`, on `bench_compat_liver_3d`: fifteen of the sixteen features
agree to the last bit (worst relative residual 1.2e-15, `3GLSZM_GLV`), asserted at `rel=1e-9`.
`3GLSZM_ZE` sits at 1.9e-4 because `calc_sums_of_P()` takes the family's only logarithm through
`fast_log10(...)/LOG10_2`; it asserts at `rel=1e-3`, the documented-residual tier.

**The cell is measured on the matrix, not only on the sixteen scalars.** Every feature of this family
is a contraction of one `P(i, j)` size-zone matrix, so two errors inside it that cancel would leave
all sixteen scalars unmoved. `test_3d_glszm_matrix_pyradiomics` pins all 186 non-empty cells of the
20×634 matrix against PyRadiomics' own `P_glszm`, plus its `Ng`, `Ns`, `Nz`, `Np` and a count that
no cell outside those 186 is populated. It reads the voxel cube the same extraction produced and
bins it with the same `bin_intensities_3d()` call `calculate()` makes, so it and the scalar
assertions describe one run.

## Why the `IBSI=true` half is one unmeasured row and not one per level count

`IBSI=true` makes `GLSZM_GREYDEPTH` inert — `calculate()` sets `greyInfo = 0` regardless of what was
passed — so that half of the cross-product collapses to a single point. **No 3D GLSZM assertion in
the tree runs on it.** It is recorded here rather than omitted, because it is not merely the same
arithmetic under another name: at `IBSI=true` the matrix row index becomes `zone.first - 1` and `Ng`
becomes `max_element(I)` rather than `I.size()`, a distinct branch in the fill loop. On a fixture
whose levels are contiguous from 1 the two branches coincide, which is why nothing has ever noticed
the difference; on one with a gap they need not. The next revisit should measure it rather than
assume it.

## `GLSZM_GREYDEPTH == 0` is a third configuration, not a variant of the other two

At MATLAB binning the background level is bin 1 (`zeroI = 1`), so a voxel binned there starts no zone
of its own; at radiomics binning and at no binning the background is intensity 0. That is a
different quantity, not a different resolution of the same one, which is why the drift guard carries
no oracle claim and why the recipes state the value explicitly instead of leaning on the zero a
settings vector starts at.

The unwired version of `test_3d_glszm_regression.h` did lean on it: it set `GREYDEPTH=64` and left
`GLSZM_GREYDEPTH` at 0, and at that setting every one of its sixteen pins is wrong by between 1.6×
and 1.5e6× (`tests/vetting/audit/glszm_3d_pyradiomics_vetting_report.md`, §"What the orphan would
have done"). The `IBSI=false, no binning` row above is that cell: reachable by a real run, and
covered today only by an is-it-finite mechanics assertion.

## The connectivity check is a fixture, not a cell

The 26-neighbourhood is not selectable, so no row of this matrix exercises it as a config point. A
4×4×3 fixture does: `glszm_3d_pyradiomics_small_volume`, one populated slice between two empty ones,
whose nine zones over four grey levels can be counted by eye and whose matrix
`test_3d_glszm_smallmatrix_pyradiomics` asserts cell by cell. The phantom cell above shows the same
property at scale, on 186 cells at once, but nobody can check those by hand — which is the division
of labour the two assertions exist for.

The small volume is the fixture rather than a copy of one: there is no file to read it from, and
`gen_glszm3d_pyradiomics.py` runs PyRadiomics on the same literal at `binWidth=1`. That is what
keeps it out of SPEC §5.2's hand-labelled self-consistency — its expected values come from an
independent tool, not from the model that wrote it.
