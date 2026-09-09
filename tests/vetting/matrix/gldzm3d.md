# 3D GLDZM config matrix

Axes are the settings `D3_GLDZM_feature` actually reads, per SPEC §5.2 step 1 — extracted from
`calculate()` / `prepare_GLDZM_matrix_kit()` in `src/nyx/features/3d_gldzm.cpp`, not from the
settings struct at large:

```cpp
auto greyInfo = STNGS_NGREYS(s);
auto greyInfo_localFeature = D3_GLDZM_feature::n_levels;
if (greyInfo_localFeature != 0 && greyInfo != greyInfo_localFeature)
    greyInfo = greyInfo_localFeature;
if (STNGS_IBSI(s))
    greyInfo = 0;
```

Two settings, but **not two independent axes**: `IBSI=true` overwrites the grey depth with 0, and
every branch below keys off `greyInfo` alone through `ibsi_grey_binning` (`== 0`),
`matlab_grey_binning` (`> 0`) and `radiomics_grey_binning` (`< 0`). So the axis is the **binning
scheme**, it has exactly three values, and `IBSI=true` and `GREYDEPTH=0, IBSI=false` are two
spellings of one of them rather than two points. Everything else in `Fsettings` is ignored by this
family.

The static `D3_GLDZM_feature::n_levels` overrides `GREYDEPTH` for this family alone when non-zero.
No test assigns it and nothing on the command line reaches it, so it is not an axis.

Two things that are axes in some GLDZM implementations are **not settable** here, and that is part of
the verdicts below rather than a gap in them: the zone connectivity is fixed at 26 and the distance
metric at the city-block distance to the ROI border, which is what IBSI defines for both.

## Config points

Three, one per binning scheme. Each is listed once, with a measured disposition and the assertions
that back it.

| # | binning scheme | how it is spelled | verdict | backed by |
|---|---|---|---|---|
| 1 | **none** — grey level = the voxel's own value | `IBSI=true` (any `GREYDEPTH`), or `GREYDEPTH=0, IBSI=false` | **VALID(mirp)** | recipe `gldzm3d.mirp_compat_phantom`: 16 features pinned against MIRP in `test_3d_gldzm_mirp.h` at an absolute 1e-9, measured residual 5.7e-15. `test_3d_gldzm_no_binning_spellings_agree_mechanics` holds the two spellings to one set of values; `test_3d_gldzm_zero_level_voxels_are_zoned_mechanics` holds the level-0 lift this point needs |
| 2 | **MATLAB** — `floor(n·i / ROI max) + 1`, clipped into `[1, n]` | `GREYDEPTH > 0, IBSI=false` | **VALID-BUT-PRODUCTION-ONLY** | recipe `gldzm3d.regression_ut_phantom`: all 18 features pinned in `test_3d_gldzm_regression.h` at `rel=1e-9` on `bench_ut57_3d`. No tool reproduces this scheme — see "Why point 2 has no oracle" |
| 3 | **radiomics** — `(i − ROI min) / binW + 1`, `binW = (max − min) / n` | `GREYDEPTH < 0, IBSI=false` | **VALID(mirp)**, on a fixture the scheme maps identically | `test_3d_gldzm_radiomics_binning_is_identity_here_mechanics`: at `GREYDEPTH=-8` on `bench_compat_gldzm_3d` the scheme is the identity on the fixture's levels 1..8, so the point returns point 1's values and MIRP's goldens for point 1 apply unchanged. Measured, not argued — see the limit below |

`3GLDZM_GLM` and `3GLDZM_ZDM` are outside every cell of this table: MIRP's GLDZM emits no
`dzm_gl_mean` / `dzm_zd_mean` column and IBSI defines neither, so no oracle in the tool matrix
reaches them at any binning scheme. They are drift guards on point 2 and stay so.

## Reference mappings, and the evidence behind the verdicts

These are **tool** configurations, not Nyxus config points, so they are not rows above. Each says how
MIRP was set to answer one of the points, and what the answer was. All three are reproduced by
`oracles/gen_gldzm3d_mirp.py`; the measurements are in
`audit/gldzm_3d_mirp_vetting_report.md`.

| mapping | MIRP settings | answers | result |
|---|---|---|---|
| `gldzm3d.mirp_compat_phantom` | `by_slice=False`, `base_discretisation_method="none"`, on `bench_compat_gldzm_3d` | point 1 | agrees to 5.7e-15 absolute; **this is what makes point 1 VALID** |
| `gldzm3d.mirp_fbn64` | `fixed_bin_number` n=64 on `bench_ut57_3d` | nothing | **not config-matched to point 2**, and cannot be made so: MIRP's `fixed_bin_number` bins over `[ROI min, ROI max]` and lands the ROI on levels 1-64, Nyxus' MATLAB scheme bins from a lower edge of 0 and lands it on 22-64, 43 distinct. The residual measures the two discretisations, not the GLDZM |
| `gldzm3d.mirp_samelevels` | `base_discretisation_method="none"` on the grey levels Nyxus produced, on `bench_ut57_3d` | point 1, at scale | agrees to 1.0e-14 over a 274,432-voxel ROI, which is what says point 1's agreement is not an artifact of the compatibility phantom's size. Not a Nyxus configuration: no command line produces it |

## Why point 2 has no oracle

Not because the GLDZM is in doubt — point 1 settles that — but because the MATLAB scheme's lower bin
edge is **0 rather than the ROI minimum**, over a volume the loader has already shifted by its own
minimum. Every `fixed_bin_number` tool bins over the ROI's range instead, so on a CT-like fixture the
two never share a level set. The shape of the `mirp_fbn64` residual says exactly that: the
grey-level-weighted features are the ones far from 1 (`LDLGLE` 0.13x, `LGLZE` 0.29x) while `GLNU`,
which sums a marginal without weighting it by grey level, is 0.94x.

That is a property of Nyxus' binning shared by every texture family, not of this one, and it is
tracked as its own backlog item. Closing it would make point 2 assertable against MIRP, and its rows
would then sit beside point 1's rather than replacing them.

## The limit on point 3

Point 3's verdict is measured on a fixture whose levels the scheme happens to map onto themselves. At
a bin count that actually re-bins, the radiomics scheme is a Nyxus convention with the same
lower-edge question point 2 has — it bins from the ROI minimum rather than 0, which is closer to what
the tools do, but the fixture to measure it on does not exist yet. What point 3's assertion
establishes is that the scheme is wired to the same zone search and distance transform as point 1 and
does not take a different path through the family; what it does not establish is agreement at a bin
count that changes the levels. Stated here rather than left to a reader of the test name.
