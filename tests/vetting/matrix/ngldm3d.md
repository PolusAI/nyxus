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

- the **neighbourhood distance** *d* is fixed by the `shifts` table in `3d_ngldm.cpp`;
- the **coarseness parameter** *alpha* is fixed at 0 (an exact grey-level match is required).

## Config points

| GREYDEPTH | IBSI | verdict | recipe / oracle |
|---|---|---|---|
| 64 | false | **VALID-prod-only** → regression | `ngldm3d.regression_ut_phantom`; all 19 features pinned in `test_3d_ngldm_regression.h` on `bench_ut57_3d` at `rel=1e-9` |
| 64 | false | **VALID(mirp) in principle, FAILING in fact** | `ngldm3d.mirp_fbn64` — MIRP at `fixed_bin_number`, n=64, d=1, alpha=0 is config-matched to the row above, and disagrees on 16 of the 17 features it computes |
| 0 (IBSI ladder) | true | **not exercised** | no 3D NGLDM assertion runs IBSI mode; the 2D family does (`ngldm.ibsi_phantom_2d`), the 3D one has no published consensus values to run against |
| any | any | **INVALID** — no such point | *d* and *alpha* cannot be varied from settings (see above), so the cross-product has no further dimensions to sweep |

## Why the measured cell is a defect rather than a convention

SPEC §5.2 says verdicts are measured, and the measurement here does not produce an agreement. The
config-matched MIRP run disagrees by factors from 0.003x to 50x, and the cause is in Nyxus and
identified — not a definitional difference that a band could honestly cover:

- `calc_ngld_matrix` iterates the ROI **bounding box** and explicitly does not skip background
  voxels (`// Do not skip off-ROI pixels`), so 551 040 box voxels enter a matrix that IBSI defines
  over the ROI's 274 432;
- the 3D neighbourhood in `shifts` has **24** members, not 26 — the two pure-axial voxels
  `(0,0,±1)` are absent, and `int maxNr = nsh + 1;` still carries a comment describing the 2D
  count of 8.

So this family has no `vetted` row, every feature is a snapshot on the first cell, and
`ngldm3d.mirp_fbn64` exists so the divergence stays reproducible and the promotion can be re-run
against a fixed implementation. Evidence: `audit/ngldm_3d_mirp_vetting_report.md`. The defect is
tracked for its own branch; pinning Nyxus' current numbers is a change detector, not an endorsement.

## What would move a cell to VALID

Fixing the two implementation defects above, then re-running `oracles/gen_ngldm3d_mirp.py`: the
recipe, the fixture and the generator are already in place, so promotion is a re-measurement rather
than new plumbing.
