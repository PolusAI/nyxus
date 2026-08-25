# NGTDM config matrix

Axes = the settings `NGTDMFeature::calculate()` actually reads; verdicts are measured, not assigned
(SPEC §5.1). There are three and they collapse to two: `calculate()` reads `STNGS_NGREYS`,
`NGTDMFeature::n_levels` and `STNGS_IBSI` (`ngtdm.cpp:44-49`). The static wins over `GREYDEPTH`
whenever it is non-zero and differs, so the two grey-count knobs are one axis with a precedence rule
rather than two; and `ibsi=true` overwrites the result with `0`, which collapses that whole half of
the cross-product to a single point.

`GREYDEPTH` carries the binning scheme in its sign (`texture_feature.h:101-103`): `0` = no binning
(IBSI), `> 0` = MATLAB binning at that level count, `< 0` = radiomics binning at `abs()` bins. The
static follows the same convention, because it is substituted for `GREYDEPTH` rather than read
separately.

**Three things that look like knobs and are not.** `PIXELDISTANCE` is never read — the neighbourhood
is the d=1 8-neighbourhood the IBSI definition fixes, so a pixel distance set in an NGTDM test looks
meaningful and changes nothing, which is why the recipes do not set one. There is no angle axis
either: NGTDM sums over the whole neighbourhood rather than per direction, so there is nothing to
aggregate and no `_AVE` feature in the family. And there is no `alpha`/difference-level axis of the
kind NGLDM has.

| ibsi | binning | levels | verdict | recipe / oracle |
|---|---|---|---|---|
| True | none (identity on an already-discrete fixture) | phantom's own 1..6 | VALID, all 5 features | `ngtdm.ibsi_phantom_2d` — mirp + ibsi, at SPEC §7's exact tier |
| False | MATLAB (`GREYDEPTH > 0` or `n_levels > 0`) | 100 | VALID-prod-only, all 5 | no oracle reproduces it → `test_2d_ngtdm_regression.h` |
| False | MATLAB | 128, from `GREYDEPTH` with the static left at its default 0 | NOT PINNED | reachable and different — see below |
| False | radiomics (`GREYDEPTH < 0`) | any | NOT MEASURED | see below |
| any | any | ROI binning down to fewer than 2 levels | INVALID | degenerate — `calculate()` returns NaN before building a matrix (`ngtdm.cpp:70-78`), guarded in code rather than by an oracle |

## Why the IBSI point is one row and not one per level count

`IBSI=true` sets the grey-binning info to `0` after the static has been substituted, so every level
count — and the static — collapse to the same point. The tests pass `GREYDEPTH=128` only because the
shared fixture takes a value, and pass `n_levels=0` only to say which value they are not depending
on. The fixture is the four IBSI phantom slices, whose grey levels are already discrete 1..6, so "no
binning" is also an identity operation on them: Nyxus and mirp assign the same levels by construction
rather than by agreement.

**The static's irrelevance here is asserted, not assumed.** `NGTDMFeature::n_levels` is a `static int`
shared by every test in the binary, and `test_2d_ngtdm_regression.h` needs it at 100.
`test_2d_ngtdm_mechanics.h` runs the IBSI point at `n_levels = 0` and at `n_levels = 100` and
requires the two to be bit-equal, which is what makes the collapse above a measurement rather than a
reading of the source. Being a mechanics test it establishes no vetting and no registry row cites it.

## Measured agreement at the VALID point

mirp 2.6.0 at `by_slice=True`, `base_discretisation_method="none"`, against Nyxus in IBSI mode, over
5 features × (4 slices + 1 four-slice mean) = 25 comparisons. Worst **absolute** residual 3.6e-15, on
`NGTDM_COMPLEXITY` slice 2; worst **relative** residual 3.2e-16, on `NGTDM_CONTRAST` slice 2 — float
summation order and nothing else, which is why `test_2d_ngtdm_mirp.h` asserts at SPEC §7's exact tier
(absolute 1e-9) rather than at the `rel=1e-3` same-definition row.

This family has no entropy term, so the `Nyxus::fast_log10` residual that costs 2D GLDM and 2D GLSZM
their exact tier on one feature each does not arise here and all five stay at the tier.

PyRadiomics 3.0.1, run at the same cell with `binWidth=1` and `force2D=True`, agrees with mirp to
1.6e-16 and with Nyxus to 2.4e-16, so the cell's verdict does not rest on one tool. Only mirp's values
are pinned; a second table identical to the first to 1.6e-16 is redundancy, not coverage.

The same four-slice means are pinned against the published IBSI consensus at `rel=1e-2` in
`test_2d_ngtdm_ibsi.h` — three significant figures is the precision those are published to, worst
residual 0.41% on `NGTDM_COARSENESS`.

**The cell is measured per slice, not only on the mean.** A four-slice mean cannot see errors in two
slices that cancel, and reaches a defect confined to one slice quartered. Both quantities are pinned,
5 means and 20 per-slice values, and the mean is averaged from the per-slice vector the same test just
asserted rather than from a second featurisation of the same four slices.

## Why the `ibsi=false` point is prod-only rather than a second oracle cell

Default mode is a genuinely different quantity, not a rescaling: the intensities are re-binned to a
fixed grey count instead of using the phantom's own levels, and `calculate()` then indexes the matrix
by position in the sorted unique-intensity vector rather than by raw grey level (`ngtdm.cpp:164-170`).
On the same four slices `NGTDM_CONTRAST` is 0.925 in IBSI mode and 3169.93 here. No reference
implementation computes that, so it takes the SPEC §5.1 VALID-but-production-only disposition: a drift
guard in `test_2d_ngtdm_regression.h` at `rel=1e-3` under recipe `ngtdm.default_fbn100`, with five
`status=regression` registry rows and no oracle claiming them. It is the mode a caller gets without
asking for IBSI compliance, which is why it is pinned at all.

## Not pinned: default mode with the static left at 0

The static only substitutes for `GREYDEPTH` when it is non-zero, so leaving it alone puts the fixture
on `GREYDEPTH = 128` instead of 100 — a different cell, not an absent one. It gives `NGTDM_CONTRAST`
6634.50 against the `n_levels = 100` pin's 3169.93, and nothing in the tree pins it. It is recorded
here rather than omitted so the next revisit knows it was seen and skipped, not overlooked. What
closed the risk for now is procedural:
the shared fixture takes the bin count as a parameter and restores the static it borrows, so no test
can leak one cell's grey count into another's.

## Not measured

The radiomics-binning variant of the production path (`GREYDEPTH < 0`, or a negative static) is
reachable and is a third level assignment. **No 2D NGTDM assertion in the tree runs on it** — every
test goes through `make_ngtdm2d_settings`, which passes `GREYDEPTH=128`, and the only static value
any of them sets is 100 — and this family's vetting did not measure it.

## No GPU axis

Unlike GLSZM and GLDZM, `NGTDMFeature::calculate()` has no `STNGS_USEGPU` branch: there is one
implementation of the 2D quantity. The fixture sets `USEGPU=false` anyway, so nothing here depends on
the build flag.
