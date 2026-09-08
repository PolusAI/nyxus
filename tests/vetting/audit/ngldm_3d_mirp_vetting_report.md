# 3D NGLDM vs MIRP — vetting report

The family's NGLD matrix and its feature formulas agree with MIRP 2.6.0 to machine precision when
both tools are given the same grey levels. Sixteen of the nineteen features are `vetted` against MIRP
on that basis, in `test_3d_ngldm_mirp.h` at recipe `ngldm3d.mirp_samelevels`.

The scope of that oracle is narrower than the family, and stated rather than implied: it covers the
NGLDM, not the discretisation. At MIRP's own `fixed_bin_number` the two tools do not land on the same
grey levels, and that gap is not the family's to close.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3) |
| Recipes | `ngldm3d.mirp_fbn64`, `ngldm3d.mirp_samelevels` |
| Fixture | the segmented phantom, `phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57 |
| MIRP config | `by_slice=False`, distance 1, difference level (alpha) 0; discretisation per recipe |
| Nyxus config | `GREYDEPTH=64`, `IBSI=false` — what `test_3d_ngldm_common.h`'s fixture sets, for both judging files |
| Generator | `tests/vetting/oracles/gen_ngldm3d_mirp.py` |
| Tolerance | `rel=1e-3` at `ngldm3d.mirp_samelevels` (SPEC 7 same-definition tier); `ngldm3d.mirp_fbn64` asserts nothing |

## The two tools discretise this ROI differently

MIRP's `fixed_bin_number` n=64 spreads the ROI over levels 1-64. Nyxus reaches its levels in two
steps, both outside `3d_ngldm.cpp`: the loader shifts every voxel by the volume minimum, which for
this CT phantom is -1024, and `to_grayscale(i, 0, ROI max, 64)` then truncates `i / (ROI max) * 64`.
The ROI occupies the upper two thirds of the shifted range, so it lands on levels **21-64, 44
distinct** — a 44-level ladder starting at 21 against MIRP's 64-level ladder starting at 1.

Two independent things produce that: the lower bin edge is 0 rather than the ROI minimum, and the
volume-wide shift moves the ROI up the scale. Neither is an NGLDM question, and no band over the
NGLDM features can absorb them, so `ngldm3d.mirp_fbn64` carries no rows. The vetted rows sit at
`ngldm3d.mirp_samelevels`, where the ladder is shared.

## Result at `ngldm3d.mirp_fbn64` -- MIRP discretises

Nyxus at `GREYDEPTH=64` against MIRP at `fixed_bin_number` n=64. This table measures the
discretisation gap above, not the NGLDM.

| feature | Nyxus | MIRP | Nyxus/MIRP |
|---|---|---|---|
| `3NGLDM_DCP` | 1 | 1 | 1 |
| `3NGLDM_LDE` | 0.15365 | 0.25594 | 0.600338 |
| `3NGLDM_HDE` | 40.6394 | 28.0738 | 1.44759 |
| `3NGLDM_LGLCE` | 0.000783908 | 0.0321849 | 0.0243564 |
| `3NGLDM_HGLCE` | 1873.25 | 1323.96 | 1.41488 |
| `3NGLDM_LDLGLE` | 7.80276e-05 | 0.000684901 | 0.113925 |
| `3NGLDM_LDHGLE` | 375.708 | 474.82 | 0.791263 |
| `3NGLDM_HDLGLE` | 0.056243 | 8.71408 | 0.00645427 |
| `3NGLDM_HDHGLE` | 44248.7 | 14942.8 | 2.96119 |
| `3NGLDM_GLNU` | 6480.48 | 4350.27 | 1.48967 |
| `3NGLDM_GLNUN` | 0.0236142 | 0.0158519 | 1.48967 |
| `3NGLDM_DCNU` | 32085.4 | 40745 | 0.787469 |
| `3NGLDM_DCNUN` | 0.116916 | 0.14847 | 0.787469 |
| `3NGLDM_GLV` | 153.096 | 350.171 | 0.437203 |
| `3NGLDM_DCV` | 14.6164 | 11.9476 | 1.22338 |
| `3NGLDM_DCENT` | 8.40569 | 8.67597 | 0.968847 |
| `3NGLDM_DCENE` | 0.00347501 | 0.00287482 | 1.20878 |

The features weighted by the grey level rather than by the dependence count carry the largest ratios,
which is the signature of a level-ladder difference: `LGLCE` sums the reciprocal square of the level
and Nyxus' smallest level is 21 where MIRP's is 1, so it reads 0.024x; `HDLGLE` divides by the same
square for the same reason. The purely dependence-side `DCNU`/`DCNUN` sit at 0.79x, and `DCP` is 1 on
both sides on any input where every voxel has a same-level neighbour.

## Result at `ngldm3d.mirp_samelevels` -- the same grey levels

MIRP handed the grey levels Nyxus bins to, with `base_discretisation_method="none"`, so both compute
the NGLDM over identical levels. The fourth column is an upper bound on the relative difference, rounded up: the residuals are float
noise, so quoting them as values would publish this build's rounding rather than the tools'
agreement, and the generator verifies each run does not exceed its bound.

| feature | Nyxus | MIRP | rel <= |
|---|---|---|---|
| `3NGLDM_DCP` | 1 | 1 | 0 |
| `3NGLDM_LDE` | 0.15365 | 0.15365 | 8e-16 |
| `3NGLDM_HDE` | 40.6394 | 40.6394 | 0 |
| `3NGLDM_LGLCE` | 0.000783908 | 0.000783908 | 0 |
| `3NGLDM_HGLCE` | 1873.25 | 1873.25 | 0 |
| `3NGLDM_LDLGLE` | 7.80276e-05 | 7.80276e-05 | 9e-16 |
| `3NGLDM_LDHGLE` | 375.708 | 375.708 | 2e-16 |
| `3NGLDM_HDLGLE` | 0.056243 | 0.056243 | 7e-16 |
| `3NGLDM_HDHGLE` | 44248.7 | 44248.7 | 0 |
| `3NGLDM_GLNU` | 6480.48 | 6480.48 | 0 |
| `3NGLDM_GLNUN` | 0.0236142 | 0.0236142 | 0 |
| `3NGLDM_DCNU` | 32085.4 | 32085.4 | 0 |
| `3NGLDM_DCNUN` | 0.116916 | 0.116916 | 0 |
| `3NGLDM_GLV` | 153.096 | 153.096 | 2e-16 |
| `3NGLDM_DCV` | 14.6164 | 14.6164 | 9e-16 |
| `3NGLDM_DCENT` | 8.40569 | 8.40569 | 5e-16 |
| `3NGLDM_DCENE` | 0.00347501 | 0.00347501 | 2e-16 |

Worst relative difference over the seventeen: **8.7e-16**. Sixteen of them are asserted in
`test_3d_ngldm_mirp.h` at `rel=1e-3`, SPEC 7's same-definition tier -- a band far wider than the
measurement, left there because it has to hold on every CI platform's float and not just this one's.
`3NGLDM_GLM` and `3NGLDM_DCM` are absent from both tables: MIRP's NGLDM emits no `gl_mean` /
`dc_mean` column, so no oracle exists for them.

**What this agreement covers.** The NGLD matrix — which voxels are centres, which neighbours count,
how dependence maps to a matrix column — and all seventeen feature formulas over that matrix. It does
not cover the discretisation, because the levels are an input to the comparison rather than a result
of it: the generator reproduces Nyxus' binning in Python to feed MIRP. That reproduction is not taken
on trust. Its agreement to 8.7e-16 across seventeen features with different sensitivities is what
confirms it, and the generator additionally checks the measured level span (21-64, 44 distinct)
against this report.

**What it discriminates, per failure mode.** Not every assertion catches every defect, and the set is
asserted rather than a representative few precisely because of that. `3NGLDM_DCP` catches **none** of
them: Nyxus hard-codes `f_DCP = 1`, so it is a constant and cannot move — which is why it is the one
comparable feature with no oracle row (below), and it is excluded from the sixteen counted here.

| an implementation that... | fails |
|---|---|
| takes NGLDM centres over the bounding box rather than the ROI | **all sixteen** — `Ns` changes from 274,432 to 511,360, and `GLNU`/`GLNUN` and `DCNU`/`DCNUN` each equal `Ns` exactly |
| visits 24 of the 26 Chebyshev-1 neighbours | the dependence-weighted ones: `LDE`, `HDE`, `LDLGLE`, `LDHGLE`, `HDLGLE`, `HDHGLE`, `DCNU`, `DCNUN`, `DCV`, `DCENT`, `DCENE` |
| reads the dependence count as the matrix column `j` rather than `j+1` | `LDE`, `HDE`, `LDLGLE`, `DCV` — but **not** `LGLCE` or `HGLCE`, which weight by grey level only |
| aggregates `GLNU` over anything but the grey-level row marginal | `GLNU`, `GLNUN` |
| takes the grey level as the row index `i+1` rather than the LUT value `U[i]` | `GLV` by 2.7x, and every grey-level-weighted feature moves |

The rows are not redundant: no single assertion in the table above catches all five, and two of the
five are caught by fewer than half the set.

## The family's own settings

`D3_NGLDM_feature` reads two: `GREYDEPTH` and `IBSI`. The neighbourhood distance *d* and the
coarseness parameter *alpha* are not settable — *d* is fixed by the `shifts` table in
`3d_ngldm.cpp` and *alpha* is fixed at 0, an exact grey-level match. Both are therefore matched to
MIRP by construction at `d1_a0.0`, and the config matrix (`matrix/ngldm3d.md`) has no further
dimensions to sweep.

`IBSI` is not a second ladder: `to_grayscale` reads it as `disable_binning`, so `IBSI=true` turns
binning off and makes the raw intensity the grey level. MIRP matches that by construction too, at
`base_discretisation_method="none"` over the raw values. Measured on this fixture it reaches Ng =
2001 distinct raw levels with **83.46% of ROI voxels having no matching neighbour** and a maximum
dependence of 17 of a possible 26 — a valid config point and a weak discriminator, recorded in the
matrix as owed an oracle round rather than claimed as one.

## Why `3NGLDM_DCP` is `regression` and not `vetted`

Dependence-count percentage is the fraction of voxels having at least one dependency. On any input
where every voxel has a same-binned neighbour it is exactly 1, which holds for this phantom under
every neighbourhood and every level ladder, and both tools report 1. It is an agreement that cannot
fail for the reasons an assertion would want it to fail, so it is left out of
`test_3d_ngldm_mirp.h` and stays a drift guard. It is the only comparable feature excluded on those
grounds; the other two without oracle rows, `3NGLDM_GLM` and `3NGLDM_DCM`, are excluded because MIRP
does not compute them.

## Include hygiene and file-level observations

The family is one header, `test_3d_ngldm_regression.h`: there is no `_common.h` and no oracle file,
and `test_3d_ngldm_coverage.h` has since been retired — see `ngldm_3d_golden_regen.md`, "The retired
coverage sweep". That makes the regression header the family's fixture as well as its table, so the
include rule below applies to it with no `_common.h` to lean on.

- Its includes are all direct: `<iomanip>`, `<tuple>` and `helpers/fsystem.h` (for `fs::exists`)
  alongside the four headers the mocked 3D workflow needs, and `<iostream>` is left to
  `test_main_nyxus.h`, which supplies it. Its golden lookup is guarded with `find()` rather than
  `operator[]`, which would default-insert a missing key as 0 and compare against a fabricated
  reference.
- **`test_3d_ngldm_coverage.h`** kept its single include of `test_3d_coverage_common.h`, as the 3D
  `_coverage.h` files do (SPEC §6.3.1). That file is now retired.
- The 19 tests in `test_3d_ngldm_regression.h` do assert, verified by negative control on two pins:
  perturbing `3NGLDM_DCENE` from 0.0035 to 0.99 fails `TEST_3D_NGLDM_DCENE_REGRESSION`, and pinning
  `3NGLDM_GLNU` at 115443.18172715895 — the value an NGLDM built over the bounding box produces —
  fails `TEST_3D_NGLDM_GLNU_REGRESSION`. The second is the one with teeth: it is the assertion that
  would catch a regression to counting background voxels as centres.

`tests/vetting/TOOLS.md` gains nothing here. The one trick worth recording — reading an uncompressed
NIfTI-1 phantom with numpy so a MIRP generator stays single-env — is recorded by the 3D morphology
PR, and `gen_ngldm3d_mirp.py` reuses it rather than duplicating the entry.

The 3D config matrix is `matrix/ngldm3d.md` (SPEC §5.1) and the benchmark `bench_ut57_3d` the 19
rows cite is defined in `benchmarks.md` (SPEC §6.3). The **2D** family still has no
`matrix/ngldm.md`, which is a gap in the 2D family's paperwork rather than this one's.

## Reproduction

```
# MIRP side, both runs (conda env with mirp 2.6.0)
python tests/vetting/oracles/gen_ngldm3d_mirp.py

# Nyxus side
runAllTests --gtest_filter=*3D_NGLDM_DUMP_REGRESSION*

# coverage artifact
python tests/vetting/audit/scan_ngldm3d_coverage.py [--check]
```
