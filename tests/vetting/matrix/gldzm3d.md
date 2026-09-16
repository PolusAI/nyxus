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

Three, one per binning scheme. Each is listed once, with a measured disposition and the rows and
assertions that back it. Every point carries coverage for **all 18 features**: the 16 MIRP reaches
as oracle rows, and `3GLDZM_GLM` / `3GLDZM_ZDM` as drift guards, because those two are Nyxus outputs
at every cell even though no tool computes them.

| # | binning scheme | how it is spelled | verdict | rows | backed by |
|---|---|---|---|---|---|
| 1 | **none** — grey level = the voxel's own value | `IBSI=true` (any `GREYDEPTH`), or `GREYDEPTH=0, IBSI=false` | **VALID(mirp)** | 16 `vetted` + 2 `regression` at `gldzm3d.mirp_compat_phantom` | the 16 pinned in `test_3d_gldzm_mirp.h` at an absolute 1e-9, measured residual 5.7e-15; `GLM`/`ZDM` pinned in `test_3d_gldzm_regression.h`. `test_3d_gldzm_no_binning_spellings_agree_mechanics` holds the two spellings to one set of values; `test_3d_gldzm_zero_level_voxels_are_zoned_mechanics` holds the level-0 lift this point needs |
| 2 | **MATLAB** — `floor(n·i / ROI max) + 1`, clipped into `[1, n]` | `GREYDEPTH > 0, IBSI=false` | **VALID-BUT-PRODUCTION-ONLY** | 18 `regression` at `gldzm3d.regression_ut_phantom` | all 18 pinned in `test_3d_gldzm_regression.h` at `rel=1e-9` on `bench_ut57_3d`. No tool reproduces this scheme — see "Why point 2 has no oracle" |
| 3 | **radiomics** — `(i − ROI min) / binW + 1`, `binW = (max − min) / n` | `GREYDEPTH < 0, IBSI=false` | **VALID(mirp)** at a bin count the scheme maps identically | 16 `vetted` + 2 `regression` at `gldzm3d.mirp_compat_phantom_radiomics` | the 16 asserted against MIRP **at this cell's own settings** in `test_3d_gldzm_mirp.h`, `GREYDEPTH=-8` on `bench_compat_gldzm_3d`; `GLM`/`ZDM` pinned beside them. `test_3d_gldzm_radiomics_binning_is_identity_here_mechanics` holds the identity that is the *reason* MIRP's goldens transfer — the coverage is the 16 rows, not the equivalence; `test_3d_gldzm_zero_level_voxels_are_zoned_radiomics_mechanics` holds the level-0 lift this point needs, which gathers its levels as a set rather than a ladder and so is its own code path |

**What point 3 does and does not establish.** At `GREYDEPTH=-8` the scheme is the identity on this
fixture's levels 1..8: `to_grayscale_radiomix` maps them over a bin width of 7/8 back onto
themselves, the top level clipping into the last bin. So the cell is vetted **at that bin count on
that fixture**. A bin count that actually re-bins is a different matter — there the scheme is a
Nyxus convention with the same lower-bin-edge question point 2 has, it bins from the ROI minimum
rather than 0, which is closer to what the tools do, and the fixture to measure it on does not exist
yet. That configuration is the one thing about this family still unexercised.

## Reference mappings, and the evidence behind the verdicts

These are **tool** configurations, not Nyxus config points, so they are not rows above. Each says how
MIRP was set to answer one of the points, and what the answer was. All three are reproduced by
`oracles/gen_gldzm3d_mirp.py`; the measurements are in
`audit/gldzm_3d_mirp_vetting_report.md`.

| mapping | MIRP settings | answers | result |
|---|---|---|---|
| `gldzm3d.mirp_compat_phantom` | `by_slice=False`, `base_discretisation_method="none"`, on `bench_compat_gldzm_3d` | point 1 | agrees to 5.7e-15 absolute; **this is what makes point 1 VALID** |
| `gldzm3d.mirp_fbn64` | `fixed_bin_number` n=64 on `bench_ut57_3d` | nothing | **A CONFIG-MISMATCHED MAPPING, not a Nyxus config point and not an invalid one.** MIRP's `fixed_bin_number` bins over `[ROI min, ROI max]` and lands the ROI on levels 1-64; Nyxus' MATLAB scheme bins from a lower edge of 0 and lands it on 22-64, 43 distinct. The two never share a level set, so the residual measures the two discretisations rather than the GLDZM, and no row can be written from it. Point 2 is a perfectly valid Nyxus configuration; what is mismatched is this attempt to map a tool onto it |
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

## What is not covered

**One thing: the radiomics scheme at a bin count that actually re-bins.** Point 3 is vetted at a bin
count the scheme maps identically, which is what makes MIRP's goldens applicable to it; a bin count
that changes the levels puts the cell back into the same lower-bin-edge question point 2 has, and
covering it needs a fixture that does not exist yet, not new plumbing.

`3GLDZM_GLM` and `3GLDZM_ZDM` are **not** in this list. They cannot be *vetted* anywhere — MIRP's
GLDZM emits no `dzm_gl_mean` / `dzm_zd_mean` column and IBSI defines neither — but they are Nyxus
outputs at every cell and every cell carries a drift guard for them, so no cell is uncovered.
