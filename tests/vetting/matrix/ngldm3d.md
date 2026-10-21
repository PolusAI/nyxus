# 3D NGLDM config matrix

Axes are the settings `D3_NGLDM_feature` actually reads, per SPEC §5.2 step 1 — extracted from
`calculate()` / `prepare_NGLDM_matrix_kit()` in `src/nyx/features/3d_ngldm.cpp`, not from the
settings struct at large:

```cpp
prepare_NGLDM_matrix_kit (NGLDM, greyLevelsLUT, Ng, Nr, r, STNGS_NGREYS(s), STNGS_IBSI(s));
```

Two, therefore: **grey binning** (`GREYDEPTH`, reaching the feature as `n_greys`, whose sign selects
the scheme as in every texture family -- positive is that many levels over `[0, ROI max]`, 0 is the raw
levels, negative is `abs()` PyRadiomics-style bins over the ROI's own range) and **IBSI mode** (`IBSI`,
which keeps the raw intensity as the grey level -- so it does not select a different ladder, it turns
binning off).

Two things that would be axes in the IBSI definition are **not settable** here, and that is part of
the dispositions below:

- the **neighbourhood distance** *d* is fixed by the `shifts` table in `3d_ngldm.cpp`, at the 26
  voxels of Chebyshev distance 1;
- the **coarseness parameter** *alpha* is fixed at 0 (an exact grey-level match is required).

Both therefore match MIRP's `d1_a0.0` by construction, and the cross-product has no further
dimensions to sweep.

## Config points

One row per Nyxus config point, verdict per SPEC §5.1.

| GREYDEPTH | IBSI | verdict | oracle / reason |
|---|---|---|---|
| 64 | false | **VALID** | `mirp` at recipe `ngldm3d.mirp_samelevels` — 16 features `vetted` in `test_3d_ngldm_mirp.h` at `rel=1e-3`. The same point carries the family's three remaining drift guards, `3NGLDM_DCP` / `3NGLDM_GLM` / `3NGLDM_DCM`, on `bench_ut57_3d` at `rel=1e-9` (`ngldm3d.regression_ut_phantom`) — the only three no tool can judge |
| any other > 0 | false | **VALID** — same cell | a positive `GREYDEPTH` sets the ladder's height, not its kind. Every positive depth reaches MIRP the way `ngldm3d.mirp_samelevels` does, so it is one cell exercised at the depth the tests pin (64), not a family of cells |
| 0 | false | **VALID** — the `IBSI=true` cell | no binning: the raw intensity is the grey level, exactly as at `IBSI=true`. `test_3d_ngldm_zero_greydepth_equals_ibsi_mechanics` holds all nineteen features equal to that run bit for bit, so the cell below covers it |
| < 0 | false | **VALID** | `mirp` at recipe `ngldm3d.mirp_fbn20_negdepth` — 16 features `vetted` in `test_3d_ngldm_mirp.h` at `rel=1e-9` (`3NGLDM_DCENT` at `rel=1e-6`), `GREYDEPTH=-20`, worst measured residual 5.1e-16. IBSI's fixed bin number over the ROI's range, which MIRP's `fixed_bin_number` computes for itself, so the binning is compared too |
| any | true | **VALID** | `mirp` at recipe `ngldm3d.mirp_ibsi_rawlevels` — 16 features `vetted` in `test_3d_ngldm_mirp.h` at `rel=1e-3`, worst measured residual 7.54e-15. `IBSI` keeps the raw level, so this point does not bin: the raw intensity is the grey level (2001 distinct here) and MIRP at `base_discretisation_method="none"` matches by construction. `GREYDEPTH` is not read when `IBSI=true`, which is why the cell spans every depth |

`d` and `alpha` add no rows: neither can be varied from settings, so there is no further cross-product
and no `INVALID` cell to record.

## The two MIRP mappings of the `IBSI=false` point

These are *mappings onto the reference tool*, not config points — the Nyxus side of both is the one
`GREYDEPTH=64, IBSI=false` run in the first row above.

| recipe | how MIRP is set | equivalent? |
|---|---|---|
| `ngldm3d.mirp_samelevels` | `base_discretisation_method="none"` over the grey levels Nyxus bins to | **Yes** — one ladder on both sides. This is what the 16 `vetted` rows assert, agreeing to a worst 8.7e-16 |
| `ngldm3d.mirp_fbn64` | `fixed_bin_number` n=64, MIRP discretising the ROI itself | **No** — MIRP puts this ROI on levels 1-64 and Nyxus on 21-64, 44 distinct. Non-equivalent, so it carries no rows; kept because it is the measurement of that gap |

Nyxus reaches its levels through the loader's shift by the volume minimum (−1024 on this CT phantom)
and then `to_grayscale(i, 0, ROI max, 64)`. The lower bin edge is 0 rather than the ROI minimum, and
the shift shifts the ROI up the ladder — neither an NGLDM question, which is why the non-equivalence
is recorded here rather than absorbed by a band.

## The two VALID cells establish different things

`ngldm3d.mirp_samelevels` (the `IBSI=false` cell) is **scope-narrowed**, per SPEC §4, and the notes
say so on the assertion, the registry row and the recipe. It covers the NGLD matrix — which voxels
are centres, which neighbours count, how a dependence count maps to a matrix column — and the sixteen
feature formulas over it. It does **not** cover the discretisation: the levels are an input to the
comparison rather than a result of it, since the generator reproduces Nyxus' binning in numpy.

`ngldm3d.mirp_ibsi_rawlevels` (the `IBSI=true` cell) is **not** narrowed, because there is no binning
step to reproduce — the raw intensity is the grey level on both sides. It is the
weaker discriminator of the two (83.46% of ROI voxels have no matching neighbour, maximum dependence
17 of 26) and the wider in scope.

`ngldm3d.mirp_fbn20_negdepth` (the negative-`GREYDEPTH` cell) is not narrowed either: MIRP bins the
ROI itself at `fixed_bin_number` n=20 and Nyxus at `GREYDEPTH=-20`, and the two agree to 5.1e-16, so
there the binning is judged as well. The positive scheme is the one binning still judged nowhere --
its lower bin edge of 0 has no tool counterpart, as the mappings above record.

## The three features with no oracle row

- **`3NGLDM_DCP`** — Nyxus hard-codes `f_DCP = 1`, and MIRP returns 1 on any input where every voxel
  has a same-level neighbour, at both config points. The agreement is real and cannot fail for the
  reasons an assertion should fail, so it is a drift guard only.
- **`3NGLDM_GLM`, `3NGLDM_DCM`** — MIRP's NGLDM emits no `gl_mean` / `dc_mean` column, so no tool
  reproduces them at any recipe.

These three are the whole of `ngldm3d.regression_ut_phantom`. The other sixteen keep no snapshot:
they are asserted against MIRP on the same fixture and config, so a second literal for the same Nyxus
run would add no path or config coverage.

## What is left

Every config point above is VALID and asserted. What remains is the non-equivalent MIRP mapping:
making Nyxus' texture binning span the ROI's own intensity range would put both tools on one ladder
at `fixed_bin_number` and make `ngldm3d.mirp_fbn64` assertable too. The recipe, the fixture and the
generator are already in place, so that is a re-measurement rather than new plumbing, and its rows
would sit beside the existing ones rather than replacing them (SPEC §1 counts vetting per assertion).

Evidence for every measurement on this page: `audit/ngldm_3d_mirp_vetting_report.md`.
