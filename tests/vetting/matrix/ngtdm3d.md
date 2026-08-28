# 3D NGTDM config matrix

Axes are the settings `D3_NGTDM_feature` actually reads, per SPEC §5.2 step 1 — extracted from
`calculate()` in `src/nyx/features/3d_ngtdm.cpp`, not from the settings struct at large:

```cpp
auto greyInfo = STNGS_NGTDM_GREYDEPTH(s);
if (STNGS_IBSI(s)) greyInfo = 0;
...
int neig_r = STNGS_NGTDM_RADIUS (s);
```

Three, therefore: **`NGTDM_GREYDEPTH`** (the family's own binning — `0` none, positive a MATLAB bin
count, negative a PyRadiomics-style bin count), **`IBSI`** (which forces `greyInfo` to 0 and makes the
level set a `0..max` ladder rather than the distinct values present), and **`NGTDM_RADIUS`** (the
Chebyshev radius of the neighbourhood). `GREYDEPTH` is read by the loader but not by this feature, so
it is not an axis.

Unlike the 3D NGLDM family, both of the quantities IBSI parameterises — the neighbourhood distance
*d* and the grey discretisation — **are** settable here, so the matrix has a real cross-product
rather than one usable cell.

## Config points

| NGTDM_GREYDEPTH | IBSI | NGTDM_RADIUS | verdict | recipe / oracle |
|---|---|---|---|---|
| 0 (no binning) | false | 1 | **VALID(pyradiomics)** | `ngtdm3d.pyradiomics_binwidth1` — all five features on `bench_compat_ngtdm_3d`, plus the 18-entry matrix, at `rel=1e-9`; measured residual **0** |
| 64 (MATLAB) | false | 1 | **VALID-prod-only** → regression | `ngtdm3d.regression_ut_phantom`; five pins in `test_3d_ngtdm_regression.h` on `bench_ut57_3d` at `rel=1e-9` |
| any | true | ≥1 | **not exercised** | IBSI mode builds the `0..max` ladder, which includes levels with `n_i = 0`; PyRadiomics deletes empty levels, so the two are not directly comparable and no assertion runs it |
| any | any | **0** | **INVALID — degenerate** | the neighbourhood is empty, no voxel is recorded as having a neighbour, `Nvc = 0` and every feature is `0/0`. This was the default until `compile_feature_settings()` was given `NGTDM_RADIUS = 1`; `test_3d_ngtdm_default_radius_mechanics` now holds that closed |
| 0 (no binning) | false | 2 | **VALID(pyradiomics)** | `ngtdm3d.pyradiomics_binwidth1_r2` — all five features on `bench_compat_ngtdm_3d` plus the six-row matrix, at `rel=1e-9`; measured residual **0** |
| any | any | ≥3 | **VALID in principle, not exercised** | the setting is honoured, and PyRadiomics' `distances` reaches any radius, but the phantom is 4×4×3: at radius 3 every voxel neighbours every other one and the matrix stops depending on the radius at all. A wider fixture, not a missing oracle |

## Why the first cell is measured VALID rather than assumed

Both sides index the same six grey levels for a reason that is a property of the fixture, not of an
agreement: `bench_compat_ngtdm_3d`'s minimum intensity is 0 and its values are integers, so
PyRadiomics' `binWidth=1` (`floor(x) − floor(min) + 1`) and Nyxus' zero-min correction (`+1` on every
level when `I[0] == 0`) both land on `1..6`. The residual is 0 in all 53 mantissa bits, and the
independent numpy reference in `oracles/gen_ngtdm3d_pyradiomics.py` reproduces the same values to
2.7e-16.

That coincidence is fixture-specific. On a fixture with a non-zero minimum, or with non-integer
intensities, the same two settings would **not** be config-matched — which is why the recipe records
the reason rather than only the settings.

## Why the MATLAB-binning cell carries no oracle

At `NGTDM_GREYDEPTH > 0` the binning is MATLAB-style, and `zeroI` becomes 1: a voxel binned into
level 1 is treated as background — it gets no matrix row of its own, but still counts towards its
neighbours' neighbourhood means. That is a different quantity from anything PyRadiomics or MIRP
computes (both treat every in-mask voxel as a matrix row), so the cell is production-only and its
pins are drift guards.

## What would move a cell

- **IBSI mode**: needs a reference that keeps empty grey levels in the matrix. PyRadiomics does not;
  MIRP's NGTDM output would have to be checked for it. The 2D family has published consensus values
  to run against, the 3D one does not.
- **radius 2**: closed. PyRadiomics' `distances` reaches it, and the cell is now asserted at
  `rel=1e-9` against `ngtdm3d.pyradiomics_binwidth1_r2`.
- **radius ≥ 3**: needs a wider fixture rather than a wider oracle. `bench_compat_ngtdm_3d` is
  4×4×3, so a Chebyshev radius of 3 already covers the whole volume from every voxel and radius 4
  computes the same matrix as radius 3. Nothing distinguishes an implementation that clamps the
  radius from one that honours it on this phantom.

## What `distances` means, which is not what it reads like

PyRadiomics' `distances` names Chebyshev **shells**, not a radius: `distances=[2]` is the 98 offsets
at Chebyshev distance exactly 2 and drops the 26 at distance 1. Nyxus' `gather_zones()` scans the
solid cube `-r..r`. The config-match for `NGTDM_RADIUS=2` is therefore `distances=[1, 2]`, and
`distances=[2]` is a different neighbourhood that produces different `s_i` on this phantom (`s_1` is
21.27 against the solid neighbourhood's 29.49). `distances_semantics_check()` in the generator
measures both readings against the same numpy neighbourhood every run, so a PyRadiomics release that
changed the convention would fail the generator rather than quietly re-pin the wrong cell.
