# 3D NGLDM config matrix

Axes are the settings `D3_NGLDM_feature` actually reads, per SPEC §5.2 step 1 — extracted from
`calculate()` / `prepare_NGLDM_matrix_kit()` in `src/nyx/features/3d_ngldm.cpp`, not from the
settings struct at large:

```cpp
prepare_NGLDM_matrix_kit (NGLDM, greyLevelsLUT, Ng, Nr, r, STNGS_NGREYS(s), STNGS_IBSI(s));
```

Two, therefore: **grey binning** (`GREYDEPTH`, reaching the feature as `n_greys`) and **IBSI mode**
(`IBSI`, which decides whether grey levels are the 1..Ng ladder or the ROI's distinct values).
Everything else in `Fsettings` is ignored by this family, so it is not an axis and the matrix stays
small.

Two things that would be axes in the IBSI definition are **not settable** here, and that is itself
part of the verdict below:

- the **neighbourhood distance** *d* is fixed by the `shifts` table in `3d_ngldm.cpp`, at the 26
  voxels of Chebyshev distance 1;
- the **coarseness parameter** *alpha* is fixed at 0 (an exact grey-level match is required).

Both therefore match MIRP's `d1_a0.0` by construction.

## Config points

| GREYDEPTH | IBSI | verdict | recipe / oracle |
|---|---|---|---|
| 64 | false | **VALID-prod-only** → regression | `ngldm3d.regression_ut_phantom`; all 19 features pinned in `test_3d_ngldm_regression.h` on `bench_ut57_3d` at `rel=1e-9` |
| 64 | false | **INVALID(mirp)** — the two tools discretise differently | `ngldm3d.mirp_fbn64` — MIRP at `fixed_bin_number` n=64 puts this ROI on levels 1-64, Nyxus on 21-64, so the comparison measures the level ladder rather than the NGLDM |
| 64 | false | **VALID(mirp), agreeing to 8.7e-16, not asserted** | `ngldm3d.mirp_samelevels` — MIRP handed the levels Nyxus bins to, `base_discretisation_method="none"`; see below for why no row is `vetted` at it |
| 0 (IBSI ladder) | true | **not exercised** | no 3D NGLDM assertion runs IBSI mode; the 2D family does (`ngldm.ibsi_phantom_2d`), the 3D one has no published consensus values to run against |
| any | any | **INVALID** — no such point | *d* and *alpha* cannot be varied from settings (see above), so the cross-product has no further dimensions to sweep |

## Why `ngldm3d.mirp_fbn64` is not a measurable cell

SPEC §5.2 asks for a measured verdict, and this cell measures something other than the family. MIRP's
`fixed_bin_number` n=64 discretises the ROI over levels 1-64. Nyxus reaches its levels through the
loader's shift by the volume minimum (-1024 on this CT phantom) and then
`to_grayscale(i, 0, ROI max, 64)`, which places the ROI on levels 21-64, 44 of them distinct. The
ratios in the vetting report track that difference: the grey-level-weighted features are the ones far
from 1, the purely dependence-side ones are near it.

Neither half of that is an NGLDM question — the lower bin edge being 0 rather than the ROI minimum,
and the volume-wide shift — so no band over these features can honestly cover the cell.

## Why `ngldm3d.mirp_samelevels` carries no `vetted` row

At identical grey levels the two tools agree on all seventeen comparable features to a worst relative
difference of 8.7e-16, which covers the NGLD matrix and every feature formula over it. The levels fed
to MIRP are reproduced from Nyxus' binning in the generator rather than measured out of Nyxus, so the
assertion would establish the NGLDM given a discretisation, not the discretisation. That is a real
and useful scope, and promoting rows at it is a registry decision this matrix records rather than
takes. Evidence and the discriminating power of the comparison:
`audit/ngldm_3d_mirp_vetting_report.md`.

`3NGLDM_GLM` and `3NGLDM_DCM` are outside every cell above: MIRP's NGLDM emits no `gl_mean` /
`dc_mean` column, so no tool reproduces them.

## What would move a cell to VALID

Making Nyxus' texture binning span the ROI's own intensity range would put both tools on the same
ladder and make `ngldm3d.mirp_fbn64` measurable directly. The recipe, the fixture and the generator
are already in place, so that promotion is a re-measurement rather than new plumbing.
