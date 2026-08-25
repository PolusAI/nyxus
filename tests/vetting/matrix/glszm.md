# GLSZM config matrix

Axes = the settings `GLSZMFeature::calculate()` actually reads; verdicts are measured, not assigned
(SPEC §5.1). There are two, and they are coupled: `calculate()` reads `STNGS_NGREYS` and
`STNGS_IBSI` (`glszm.cpp:83-87`) and overwrites `greyInfo` with `0` when `ibsi=true`, so the
`ibsi=true` half of the cross-product collapses to a single point. `STNGS_USEGPU` selects a second
implementation of the same quantity rather than a different quantity, so it is a build axis and not
a config one — see below.

`GREYDEPTH` carries the binning scheme in its sign (`texture_feature.h:101-103`): `0` = no binning
(IBSI), `> 0` = MATLAB binning at that level count, `< 0` = radiomics binning at `abs()` bins.

**Two things that look like knobs and are not.** Zone connectivity is fixed at 8-connected in the
zone scan, which is what IBSI defines — it is not settable, so it is not a matrix axis. Zone *size*
has no parameter either: a zone is a maximal connected set of equal-valued pixels, so unlike GLCM,
GLRLM or NGLDM this family has no distance, angle or alpha axis at all, and there is nothing left to
configure once the discretisation is fixed.

| ibsi | binning | levels | verdict | recipe / oracle |
|---|---|---|---|---|
| True | none (identity on an already-discrete fixture) | phantom's own 1..6 | VALID, 15 features | `glszm.ibsi_phantom_2d` — mirp + ibsi, at SPEC §7's exact tier |
| True | none | phantom's own 1..6 | VALID, 1 feature at a wider band | `GLSZM_ZE` — same oracles, `rel=4e-3`, the `fast_log10` residual |
| False | MATLAB (`GREYDEPTH > 0`) | 64 | VALID-prod-only, all 16 | no oracle reproduces it → `test_2d_glszm_regression.h` |
| False | radiomics (`GREYDEPTH < 0`) | any | NOT MEASURED | see below |
| any | any | ROI with `aux_min == aux_max` | INVALID | degenerate — a blank ROI has no zones; `calculate()` returns NaN before building a matrix, guarded in code rather than by an oracle |

## Why the IBSI point is one row and not one per level count

`IBSI=true` makes `GREYDEPTH` inert, so every level count collapses to the same point; the tests
pass 128 only because the shared fixture takes a value. The fixture is the four IBSI phantom slices,
whose grey levels are already discrete 1..6, so "no binning" is also an identity operation on them:
Nyxus and mirp assign the same levels by construction rather than by agreement.

## Measured agreement at the VALID point

mirp 2.6.0 at `by_slice=True`, `base_discretisation_method="none"`, against Nyxus in IBSI mode, over
16 features × (4 slices + 1 four-slice mean) = 80 comparisons. Fifteen features agree within a worst
**absolute** residual of 7.1e-15 and a worst **relative** residual of 2.0e-16, both on `GLSZM_ZV`,
slice 3 — float summation order and nothing else, which is why `test_2d_glszm_mirp.h` asserts those
at SPEC §7's exact tier (absolute 1e-9) rather than at the `rel=1e-3` same-definition row.

`GLSZM_ZE` is the sixteenth and misses by 2.5e-3 per slice, 2.0e-3 on the mean. That is not a cell
property: PyRadiomics 3.0.1, run at the same cell, agrees with mirp to 4.6e-16 and Nyxus misses it
by the same 2.5e-3, so the residual is `Nyxus::fast_log10` rather than a mirp convention. The cell
stays VALID with that one feature banded at `rel=4e-3` and the reason stated, rather than the whole
cell being loosened to fit it.

The same four-slice means are pinned against the published IBSI consensus at `rel=1e-2` in
`test_2d_glszm_ibsi.h` — three significant figures is the precision those are published to, worst
residual 0.42% on `GLSZM_LAHGLE`.

**The cell is measured per slice, not only on the mean.** A four-slice mean cannot see errors in two
slices that cancel, and reaches a defect confined to one slice quartered. Both quantities are pinned,
16 means and 64 per-slice values.

## Why the `ibsi=false` point is prod-only rather than a second oracle cell

Default mode is a genuinely different quantity, not a rescaling: `calculate()` indexes the
grey-level-dependent features by the raw intensity where IBSI mode indexes them by the grey-level
index, so on the same four slices `GLSZM_HGLZE` is 16.44 in IBSI mode and 1497.57 here. No reference
implementation computes that, so it takes the SPEC §5.1 VALID-but-production-only disposition: a
drift guard in `test_2d_glszm_regression.h` at `rel=1e-3`, with no oracle row claiming it. It is the
mode a caller gets without asking for IBSI compliance, which is why it is pinned at all.

## Not measured

The radiomics-binning variant of the production path (`GREYDEPTH < 0`) is reachable and is a third
level assignment. **No 2D GLSZM assertion in the tree runs on it** — every test in
`test_2d_glszm_{ibsi,mirp,regression}.h` goes through `make_glszm2d_settings`, which passes a
non-negative depth — and this family's vetting did not measure it. It is recorded here as an open
point rather than omitted, so the next revisit knows it was never triaged rather than assuming it
was ruled out.

## The GPU path is a build axis, not a config cell

`calculate()` branches on `STNGS_USEGPU` into `NyxusGpu::GLSZMfeature_calc`, which is compiled only
under `USE_GPU`. It is meant to compute the same quantity from the same zone matrix, so it does not
add a config point; what it needs is a same-input equality check against the CPU path, which this
family does not have today. The fixture here sets `USEGPU=false`, so every measurement above is the
CPU path and says nothing about the GPU one.
