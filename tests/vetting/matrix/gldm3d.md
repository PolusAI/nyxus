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
| `0` — **the compiled default** | false | `bench_ut57_3d` | **VALID-prod-only** → regression | `gldm3d.regression_ut_phantom_nobinning`; all 14 pinned in `test_3d_gldm_regression.h` at `rel=1e-9`; no oracle claim — see below |
| any | true | — | **same cell as the row above** | forced to `greyInfo = 0`, i.e. the same code path, reached a different way; the pins above are what guard it |
| any | any | `bench_constant_roi_3d` | **DEFECT** → regression guard | nonempty constant-intensity ROI. The intercept below fires at every binning, so the cell spans the column; asserted at `0`, where it is the only cause — `gldm3d.regression_constant_roi`, see below |

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
out `2 * (in-slice matches) + 2`. It runs that check against two tallies — the one
`D3_GLDM_feature::gather_dependence_zones()` produces, which is the traversal `calculate()` fills `P`
from, and one written out in the test file independently of it — so the neighbourhood under
assertion is production's own. `gen_gldm3d_pyradiomics.py` rebuilds the whole matrix from the
definition and requires all 163 cells to match PyRadiomics' C extension.

## The default cell

`GLDM_GREYDEPTH = 0` is not one configuration among several — it is **the compiled default**.
`Environment::compile_feature_settings()` (`src/nyx/env_features.cpp`) zero-fills every settings
vector and then names a default for `GLCM_OFFSET`, `GLCM_GREYDEPTH`, `GLCM_NUMANG` and the `FPIMG_*`
group, and for nothing else; `GLDM_GREYDEPTH` is left at the zero-fill, and only
`--3gldm/greydepth=N` moves it. So every run that does not pass that option computes this family
here, and `IBSI=true` reaches the same path deliberately. Under SPEC §5.1 that makes it a real
default config no external tool is known to reproduce — **VALID-BUT-PRODUCTION-ONLY**, i.e. a
regression row, not an unmeasured cell.

The `64` snapshot cannot stand in for it. It is a genuinely different quantity, not a rounding — on
`bench_ut57_3d` the same ROI gives:

| feature | `GLDM_GREYDEPTH=64` | `GLDM_GREYDEPTH=0` |
|---|---|---|
| `3GLDM_HGLE` | 1957.1946 | 4275550.79 |
| `3GLDM_GLV` | 153.0947 | 341996.30 |
| `3GLDM_SDE` | 0.15361 | 0.86996 |
| `3GLDM_DE` | 8.40347 | 11.48909 |

All fourteen are pinned at this cell in `test_3d_gldm_regression.h`
(`test_3d_gldm_<feature>_nobinning_regression`, recipe `gldm3d.regression_ut_phantom_nobinning`,
`rel=1e-9`), regenerated by `test_3d_gldm_dump_nobinning_regression()`. Those pins claim **no
vetting**: the oracle question — whether PyRadiomics can be made to skip discretisation on a
non-integer intensity volume — is still unanswered, so what the row establishes is that the default
configuration cannot move silently, which is what it had been able to do.

## The degenerate cell: a nonempty constant-intensity ROI

`D3_GLDM_feature::calculate()` opens with

```cpp
// intercept blank ROIs
if (r.aux_min == r.aux_max)
    ... = STNGS_NAN(s);
```

The comment says *blank*, but the test is on the intensity extrema, so it also takes a **nonempty ROI
of a single intensity** — reachable wherever a segmentation lands on a flat or saturated region. All
fourteen features come back as the configured no-value sentinel, which production defaults to `0.0`
(`ResultOptions::noval_`), a value several of the fourteen also compute legitimately.

Such an ROI has a perfectly well-defined dependence matrix: one grey level, and a dependence per
voxel. PyRadiomics 3.0.1 builds it — on a 4×4×3 cube of one intensity at `binWidth=1`, `gldm_a=0`,
`distances=[1]` (`bench_constant_roi_3d`), `P_gldm` is 1×4 = `[8, 20, 16, 4]`.

**The arithmetic beneath the intercept already agrees with that oracle exactly.** Restricting the
intercept to empty cubes (`aux_min == aux_max && aux_image_cube.size() == 0`) as a negative control,
at this cell's own configuration, Nyxus computes:

| feature | Nyxus, intercept bypassed | PyRadiomics 3.0.1 | |
|---|---|---|---|
| `3GLDM_SDE` | 0.0066408036122542298 | 0.0066408036122542298 | to the last bit |
| `3GLDM_LDE` | 239.41666666666666 | 239.41666666666666 | to the last bit |
| `3GLDM_DN` | 15.333333333333334 | 15.333333333333334 | to the last bit |
| `3GLDM_DNN` | 0.31944444444444442 | 0.31944444444444442 | to the last bit |
| `3GLDM_DV` | 26.743055555555554 | 26.743055555555554 | to the last bit |
| `3GLDM_GLN` | 48 | 48 | to the last bit |
| `3GLDM_GLV` | 0 | 0 | to the last bit |
| `3GLDM_DE` | 1.7813958326975503 | 1.7841591278514204 | 2.8e-3, `fast_log10` |
| `3GLDM_LGLE` | 0.020408163265306121 | 1 | 1/7² against 1/1²: level, not defect |
| `3GLDM_HGLE` | 49 | 1 | 7² against 1²: level, not defect |

Seven of the fourteen match to the last bit; `3GLDM_DE` carries the family's documented `fast_log10`
residual; the four grey-level features and their two cross-terms differ only because no-binning keeps
the raw intensity 7 as the grey level where PyRadiomics' `binWidth=1` maps it to level 1. So nothing
under the intercept is wrong. **The intercept is the whole defect**, and what production emits today
is the no-value sentinel — `--noval`, default `0.0` — for all fourteen.

**A fix is still not one line, because the other binning schemes fail differently.** At radiomics
binning `TextureFeature::to_grayscale_radiomix()` computes `binW = (max - min) / binCount`, which is
zero here: every voxel bins to `0`, no dependence zone is gathered, and the `Nz == 0` branch returns
the same sentinel by a second route. That path also divides by zero and converts the resulting NaN to
`PixIntens`, which the intercept is currently what keeps unreachable — so narrowing the intercept
without touching the binning would trade a wrong answer for undefined behaviour. That is why this
cell is asserted at `GLDM_GREYDEPTH=0`, where the intercept stands alone.

It is guarded rather than fixed here: `test_3d_gldm_constant_roi_regression()` pins the emitted value
for all fourteen, under a `--noval` sentinel no GLDM feature can take, so a zero-filled buffer cannot
satisfy it. Changing this moves production output for the family, and the identical
`aux_min == aux_max` intercept is shared by `3d_gldzm`, `3d_glszm`, `3d_ngldm`, `3d_ngtdm` and their
2D twins — whether each is protecting a real division is a per-family question — so it belongs on its
own branch rather than in a vetting pass.

## What would move a cell to VALID

For `GLDM_GREYDEPTH = 0` / `IBSI=true`: a PyRadiomics run with binning disabled on the same ROI. The
fixture, the generator and the recipe machinery are all in place, so it is a re-measurement rather
than new plumbing — the open part is whether the two tools mean the same thing by "no discretisation"
on a non-integer intensity volume.

For `bench_constant_roi_3d`: nothing measured is missing, and nothing needs porting — the seven
dependence-axis features already reproduce PyRadiomics to the last bit once the intercept is out of
the way. What stands between the cell and VALID is a production change: the `aux_min == aux_max`
intercept has to stop firing for nonempty ROIs, and the zero-range radiomics binning underneath has
to put a constant ROI on one foreground level instead of on background.
