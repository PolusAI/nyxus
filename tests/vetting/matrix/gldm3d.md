# 3D GLDM config matrix

Axes are the settings `D3_GLDM_feature` actually reads, per SPEC §5.2 step 1 — extracted from
`calculate()` in `src/nyx/features/3d_gldm.cpp`, not from the settings struct at large:

```cpp
auto greyInfo = STNGS_GLDM_GREYDEPTH(s);
if (STNGS_IBSI(s))
    greyInfo = 0;
bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);
```

**One and a half axes, not two.** `GLDM_GREYDEPTH` is the family's own binning, and its *sign*
selects the scheme rather than its magnitude alone:

| `GLDM_GREYDEPTH` | scheme (`texture_feature.h:101-103`) |
|---|---|
| `< 0` | radiomics — magnitude is the bin count |
| `> 0` | MATLAB — magnitude is the level count |
| `= 0` | IBSI — no binning, the ROI's own levels |

`IBSI=true` does not add a dimension: it **forces `greyInfo` to 0**, so it collapses onto the third
row above rather than crossing with it. That is why the table below is a list of points and not a
cross-product. `GREYDEPTH` is read by other families but never by this one, so a recipe that names it
is recording an inert setting, not a config point.

Three things that would be axes in the IBSI definition are **not settable** here, and that is itself
part of the verdict:

- the **neighbourhood distance** *d* is fixed at Chebyshev 1 by the 26-entry `shifts` table;
- the **coarseness parameter** *alpha* is fixed at 0 — `pi == neig_pi`, an exact grey-level match;
- there is **no angle axis**. A dependence count sums the whole neighbourhood at once, so unlike GLCM
  and GLRLM this family has no per-direction values and no aggregation step.

There is no GPU axis either: `D3_GLDM_feature::calculate()` has no `STNGS_USEGPU` branch.

## Config points

| `GLDM_GREYDEPTH` | `IBSI` | benchmark | verdict | recipe / oracle |
|---|---|---|---|---|
| `-20` (binCount 20) | false | `bench_compat_liver_3d` | **VALID(pyradiomics)** | `gldm3d.pyradiomics_bincount20`; all 14 features asserted in `test_3d_gldm_pyradiomics.h` at `abs=1e-9`, `3GLDM_DE` at `abs=4e-3` |
| `64` (MATLAB levels) | false | `bench_ut57_3d` | **VALID-prod-only** → regression | `gldm3d.regression_ut_phantom`; all 14 pinned in `test_3d_gldm_regression.h` at `rel=1e-9`; no oracle claim |
| `0` | false | — | **reachable, unmeasured** | see below |
| any | true | — | **reachable, unmeasured** | forced to `greyInfo = 0`, i.e. the same code path as the row above, reached a different way |

## The measured cell

PyRadiomics 3.0.1 and Nyxus agree at the first cell without a convention allowance, which is what
makes SPEC §7's exact tier honest here rather than generous:

| | worst | where |
|---|---|---|
| absolute | 7.11e-15 | `3GLDM_SDHGLE` |
| relative | 8.79e-16 | `3GLDM_LDLGLE` |

Eight of the fourteen agree to the last bit; the other six carry a residual of at most 7.11e-15. The band is `abs=1e-9`, six orders inside the worst
measurement. `3GLDM_DE` is the single exception at `abs=4e-3` against a measured `1.7512e-3`, for
`fast_log10()` — a documented fast path in this codebase, banded rather than reported.

Three facts make the cell comparable at all, and all three are asserted rather than assumed:
`distances=[1]` against the 26-entry `shifts` table, `gldm_a=0` against `pi == neig_pi`, and the
dependence offset — PyRadiomics puts a voxel with no dependent neighbour in column `j=1`, Nyxus
starts `nd = 1` — so the two index the same column. `TEST_3D_GLDM_SMALLMATRIX_PYRADIOMICS` is the
direct check: a 4×4×3 volume with two identical populated slices, where every dependence must come
out `2 * (in-slice matches) + 2`, and `gen_gldm3d_pyradiomics.py` rebuilds the whole matrix from the
definition and requires all 163 cells to match PyRadiomics' C extension.

## The two unmeasured cells

`GLDM_GREYDEPTH = 0` is **reachable in production**: it is what any settings vector that never names
the family's own binning gets, and `IBSI=true` reaches the same path deliberately. It is a genuinely
different quantity, not a rounding of the others — on `bench_ut57_3d` the same ROI gives:

| feature | `GLDM_GREYDEPTH=64` | `GLDM_GREYDEPTH=0` |
|---|---|---|
| `3GLDM_HGLE` | 1957.1946 | 4275550.79 |
| `3GLDM_GLV` | 153.0947 | 341996.30 |
| `3GLDM_SDE` | 0.15361 | 0.86996 |
| `3GLDM_DE` | 8.40347 | 11.48909 |

Nothing in the tree asserts it. It is recorded here rather than closed because pinning it would be a
snapshot of an unvetted third configuration, and the oracle question for it — whether PyRadiomics can
be made to skip discretisation on this fixture — has not been answered.

## What would move a cell to VALID

For `GLDM_GREYDEPTH = 0` / `IBSI=true`: a PyRadiomics run with binning disabled on the same ROI. The
fixture, the generator and the recipe machinery are all in place, so it is a re-measurement rather
than new plumbing — the open part is whether the two tools mean the same thing by "no discretisation"
on a non-integer intensity volume.
