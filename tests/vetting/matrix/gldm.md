# GLDM config matrix

Axes = the settings `GLDMFeature::calculate()` actually reads; verdicts are measured, not assigned
(SPEC §5.1). GLDM has fewer knobs than GLCM: there is no angle set, no offset and no alpha/distance
parameter. The dependence of a pixel is fixed at "self plus the 8-neighbours sharing its grey level
exactly", i.e. IBSI's alpha=0, d=1, so the only axes are the IBSI flag and the grey-binning scheme,
and the two are coupled — `calculate()` forces `greyInfo = 0` when `IBSI=true`.

`GREYDEPTH` carries the binning scheme in its sign (`texture_feature.h:101-103`): `0` = no binning
(IBSI), `> 0` = MATLAB binning at that level count, `< 0` = radiomics binning at `abs()` bins.

| ibsi | binning | levels | verdict | recipe / oracle |
|---|---|---|---|---|
| True | none (identity on an already-discrete fixture) | phantom's own 1..6 | VALID | `gldm.ibsi_phantom_2d` — pyradiomics + ibsi |
| False | MATLAB | any (64 and 128 both exercised) | VALID-prod-only | no tool reproduces the re-binning — `test_2d_gldm_regression.h` (cat2500, 128) |
| False | radiomics (`GREYDEPTH < 0`) | any | NOT MEASURED | see below |
| any | any | ROI with `Nz == 0` | INVALID | degenerate, no dependence zones — guarded in code, not an oracle point |

## Why the production point is prod-only, measured

At `IBSI=false` Nyxus re-bins the ROI with the MATLAB scheme at `coarse_gray_depth`, while
PyRadiomics bins at `binWidth=1` over the values it is handed. The two therefore build their
dependence matrices over different level assignments, and the grey-level-weighted features diverge
most because each multiplies by its own notion of the level. Measured on the canonical 154-px ROI at
`coarse_gray_depth=64`: `GLDM_SDLGLE` **108%**, `GLDM_LGLE` **96%**, `GLDM_LDLGLE` **71%**, `GLDM_DV`
**24%**. Per SPEC §5 a tolerance cannot absorb a configuration mismatch, so this point is a drift
guard and not an oracle assertion at any band. The full table is in
`audit/gldm_2d_pyradiomics_vetting_report.md`.

## Why the IBSI point is a single row

`IBSI=true` makes `GREYDEPTH` inert — `calculate()` overwrites it with `0` — so the ibsi=true half of
the cross-product collapses to one point rather than one point per level count. That is why this
matrix has four rows where GLCM's has more: the axis exists in the settings but not in the
behaviour.

## Not measured

The radiomics binning path (`GREYDEPTH < 0`) is reachable and is a third distinct level assignment,
but no 2D GLDM assertion in the tree runs on it and this family's vetting did not measure it. It is
recorded here as an open point rather than omitted, so that the next revisit knows it was never
triaged rather than assuming it was ruled out.
