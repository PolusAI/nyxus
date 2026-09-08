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

Two, therefore: **grey binning** (`GREYDEPTH`, whose sign selects the scheme — positive is a
MATLAB-style level count, negative a PyRadiomics-style bin count, 0 no binning) and **IBSI mode**
(`IBSI`, which forces the first to 0 whatever was passed). They are not independent: `IBSI=true`
collapses the grey-binning axis to its 0 point. Everything else in `Fsettings` is ignored by this
family, so it is not an axis and the matrix stays small.

The static `D3_GLDZM_feature::n_levels`, when non-zero, overrides `GREYDEPTH` for this family alone.
No test assigns it and nothing on the command line reaches it, so it is not a config point.

Two things that are axes in some GLDZM implementations are **not settable** here, and that is part
of the verdict below rather than a gap in it:

- the **zone connectivity** is fixed at 26, which is what IBSI defines in 3D;
- the **distance metric** is fixed at the city-block distance to the ROI border, likewise.

## Config points

| GREYDEPTH | IBSI | verdict | recipe / oracle |
|---|---|---|---|
| any | true | **VALID(mirp)** → vetted | `gldzm3d.mirp_compat_phantom`; 16 of the 18 features pinned in `test_3d_gldzm_mirp.h` on `bench_compat_gldzm_3d` at an absolute 1e-9, measured residual 5.7e-15 |
| 64 | false | **VALID-prod-only** → regression | `gldzm3d.regression_ut_phantom`; all 18 features pinned in `test_3d_gldzm_regression.h` on `bench_ut57_3d` at `rel=1e-9` |
| 64 | false | **INVALID** as a vetting cell | `gldzm3d.mirp_fbn64` — MIRP at `fixed_bin_number` n=64 is NOT config-matched to the row above, because the two tools put this ROI on different grey levels (1-64 against 22-64, 43 distinct). The residual measures the binning, not the GLDZM |
| negative (PyRadiomics-style bins) | false | **not exercised** | no 3D GLDZM assertion runs the radiomics binning branch; `bin_intensities_3d` reaches it and no fixture in the family does |

`gldzm3d.mirp_samelevels` is not a row above because it is not a configuration: it hands MIRP the
grey levels Nyxus produced, which no Nyxus command line has a counterpart for. It is a measurement,
recorded in the audit report, and what it establishes is that the agreement in the first row holds
on a 274,432-voxel ROI too (worst rel 1.0e-14).

## Why the vetted cell is the IBSI one

The comparison a reader would expect — MIRP's `fixed_bin_number` against Nyxus' `GREYDEPTH`, on the
same CT-like fixture — cannot be made honest, and the reason has nothing to do with the GLDZM.
Nyxus bins `floor(n * i / ROI max) + 1` over a volume the loader has already shifted by its
*minimum*, so on `bench_ut57_3d` the ROI reaches levels 22-64 while MIRP's `fixed_bin_number`
spreads the same ROI over 1-64. Every Nyxus texture family bins this way, so this is not a GLDZM
finding; it is why the vetted cell is the one where neither tool discretises at all.

`bench_compat_gldzm_3d` exists for that: its voxel values *are* the grey levels 1..8, MIRP is run
with `base_discretisation_method="none"`, and Nyxus at `IBSI=true` does no binning. Nothing about
that comparison depends on the two tools agreeing about discretisation.

## What is not covered, and what would cover it

- **The radiomics-binning branch.** `GREYDEPTH` negative reaches
  `TextureFeature::to_grayscale_radiomix`, which unlike the MATLAB branch leaves intensity 0 at
  level 0. No 3D GLDZM assertion runs it. Covering it needs a fixture and a recipe, not new
  plumbing — the family's mask now separates the ROI from the background whatever the binning does
  with zero, which is the property that branch would be exercising.
- **`3GLDZM_GLM` and `3GLDZM_ZDM`** cannot be vetted at any cell of this matrix: MIRP's GLDZM emits
  no `dzm_gl_mean` / `dzm_zd_mean` column and IBSI defines neither, so no oracle in the tool matrix
  reaches them. They are drift guards on the second row and stay so.
