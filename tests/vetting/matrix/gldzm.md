# GLDZM config matrix

Axes = the settings `GLDZMFeature::calculate()` actually reads; verdicts are measured, not assigned
(SPEC §5.1). The list is short, and shorter than it looks: `calculate()` passes exactly two settings
down to `prepare_GLDZM_matrix_kit()` — `STNGS_NGREYS` and `STNGS_IBSI` (`gldzm.cpp:381`) — and the
two are coupled, because `prepare_GLDZM_matrix_kit()` overwrites `greyInfo` with `0` when
`ibsi=true`.

`GREYDEPTH` carries the binning scheme in its sign (`texture_feature.h:101-103`): `0` = no binning
(IBSI), `> 0` = MATLAB binning at that level count, `< 0` = radiomics binning at `abs()` bins. The
static `GLDZMFeature::n_levels`, when non-zero, overrides `GREYDEPTH` for this family alone; no test
sets it, so it is not an axis here.

**Two things that look like knobs and are not.** Zone connectivity is fixed at 8-connected in
`prepare_GLDZM_matrix_kit()`, which is what IBSI defines and what GLSZM uses for the same notion of
a zone — it is not settable, so it is not a matrix axis, and the only way to test it is to pick a
fixture where 4- and 8-connected labelling disagree (see below). Zone *distance* likewise has no
parameter: it is the distance from a pixel to the nearest non-ROI pixel or image margin
(`dist2border()`), a property of the mask rather than a setting. GLDZM therefore has no distance,
angle or alpha axis at all, unlike GLCM, GLRLM or NGLDM.

| ibsi | binning | levels | verdict | recipe / oracle |
|---|---|---|---|---|
| True | none (identity on an already-discrete fixture) | phantom's own 1..6 | VALID | `gldzm.ibsi_phantom_2d` — mirp + ibsi |
| True | none | phantom's own 1..6 | VALID-prod-only, 2 features | `GLDZM_GLM`, `GLDZM_ZDM` — no oracle exposes either → `test_2d_gldzm_regression.h` |
| False | MATLAB (`GREYDEPTH > 0`) | any | NOT MEASURED | see below |
| False | radiomics (`GREYDEPTH < 0`) | any | NOT MEASURED | see below |
| any | any | ROI with `aux_min == aux_max` | INVALID | degenerate — a blank ROI has no zones; `calculate()` returns NaN before building a matrix, guarded in code rather than by an oracle |

## Why the IBSI point is one row and not one per level count

`IBSI=true` makes `GREYDEPTH` inert — `prepare_GLDZM_matrix_kit()` sets `greyInfo = 0` regardless of
what was passed — so the `ibsi=true` half of the cross-product collapses to a single point. The
fixture is the four IBSI phantom slices, whose grey levels are already discrete 1..6, so "no
binning" is also an identity operation on them: Nyxus and mirp assign the same levels by
construction rather than by agreement.

## Measured agreement at the VALID point

mirp 2.6.0 at `by_slice=True`, `base_discretisation_method="none"`, against Nyxus in IBSI mode, over
16 features × (4 slices + 1 four-slice mean) = 80 comparisons: worst **absolute** residual 1.3e-15
(`GLDZM_ZDE`, slice 2), worst **relative** residual 7.0e-16 (`GLDZM_ZDE`, slices 3 and 4). That is
float summation order and nothing else, which is why `test_2d_gldzm_mirp.h` asserts at SPEC §7's
exact tier (absolute 1e-9) rather than at the `rel=1e-3` same-definition row. The same four-slice
means are pinned against the published IBSI consensus at `rel=1e-2` in `test_2d_gldzm_ibsi.h` —
three significant figures is the precision those are published to, worst residual 0.35% on
`GLDZM_LDE`.

**The cell is measured per slice, not only on the mean.** A four-slice mean cannot see errors in two
slices that cancel, and this family's connectivity defect had exactly that shape on a neighbouring
fixture. Both quantities are pinned.

## Why `GLDZM_GLM` and `GLDZM_ZDM` are prod-only at the same cell

They sit at the VALID cell but have no oracle: neither is an IBSI GLDZM feature, so the published
consensus has no entry, and mirp exposes no column for a grey-level mean or a zone-distance mean.
They are Nyxus conventions computed from an otherwise-vetted matrix, so they are drift guards in
`test_2d_gldzm_regression.h` at `rel=1e-3` and carry `status=regression` rows. This is the SPEC §5.1
VALID-but-production-only disposition landing on two features rather than on a whole config point.

## The connectivity check is a fixture, not a cell

8-connectivity is not selectable, so no row of this matrix exercises it as a config point. Two
fixtures do, at different resolutions. The IBSI figure-3.17 worked example
(`ibsi_fig3_17a_gldzm_sample_image_*` in `test_data.h`, asserted by
`TEST_NYXUS.TEST_2D_GLDZM_MATRIX_CORRECTNESS_IBSI`) localises it: on that 4×4 ROI grey level 2 forms
one 8-connected zone and two 4-connected ones, so a single matrix cell separates the two labellings.
The phantom slices show it at feature level, on all 18 features at once, by 1.3% to 60%.

**Neither fixture was the problem; the tolerance was.** Both were already in the tree when the
defect shipped. `test_2d_gldzm_ibsi.h` asserted the four-slice means against published values at
±50%, and the largest error among those entries was 30% (`GLDZM_LDLGLE`), so the band swallowed all
fourteen. The two features that were further out — `GLDZM_ZDV` at 60% and `GLDZM_SDLGLE` at 5.3% —
were not being compared to IBSI at all: their table entries were 17-digit snapshots of Nyxus' own
pre-fix output, so they agreed with the defect by construction. Adding config points would not have
caught either half. SPEC §7's rule is what applies: a tolerance loose enough to pass a known-bad
value is a bug in the test, and per SPEC §6.3.1 a table named `_ibsi` may not hold snapshots.

## Not measured

The production path (`IBSI=false`) is reachable for 2D GLDZM and is a genuinely different level
assignment, in two variants: MATLAB binning at `coarse_gray_depth` and radiomics binning at
`GREYDEPTH < 0`. **No 2D GLDZM assertion in the tree runs on either** — every test in
`test_2d_gldzm_{ibsi,mirp,regression}.h` calls `make_gldzm2d_settings(true)` — and this family's
vetting did not measure them. They are recorded here as open points rather than omitted, so the next
revisit knows they were never triaged rather than assuming they were ruled out. Note this is a
weaker position than GLDM's, which at least drift-pins its MATLAB-binned production point.
