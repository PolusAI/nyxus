# 3D GLRLM vs PyRadiomics — vetting report

Closes the two `3GLRLM_*_AVE` rows that read `status=vetted` with no in-tree oracle assertion,
corrects the fourteen whose registry entry never recorded the assertion that already covered them,
and regenerates a regression table that no configuration reproduced.

## Tool and configuration

| | |
|---|---|
| Tool | PyRadiomics 3.0.1 / SimpleITK 2.3.1 (`binCount: 20`, `interpolator: sitkBSpline`, `weightingNorm:` empty, `imageType: Original`) |
| Recipe | `glrlm3d.pyradiomics_bincount20` |
| Fixture | the compat phantom, `compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1 |
| Nyxus config | `GREYDEPTH=100`, `IBSI=false`, `GLRLM_GREYDEPTH=-20` (negative activates binCount binning) |
| Generator | `tests/vetting/oracles/gen_glrlm3d_pyradiomics.py` |
| Tests | `test_3d_glrlm_pyradiomics.h`, `tests/python/test_nyxus.py::test_3d_glrlm_compatibility` |
| Tolerance | `rel=1e-9`, and `rel=5e-3` for `3GLRLM_RE` / `3GLRLM_RE_AVE` |

## The goldens reproduce bit-for-bit

The family had **no generator** — `gen_glrlm_{mirp,pyradiomics}.py` are 2D-only — so SPEC §6.4's
"generator script path" was unmet and the sixteen pinned numbers rested on a CLI recipe written into
a comment. `gen_glrlm3d_pyradiomics.py` now runs PyRadiomics against the compat phantom and
re-verifies every pin: **all 16 reproduce at `rel=0`**, so the table's origin is confirmed and nothing
needed re-pinning. That is the opposite outcome from 3D GLCM, where the same exercise found `IDN` and
`IDMN` pinned to byte-identical values — and it is only knowable by running the tool.

The generator also checks the oracle's own output against the five bounds the family carries by
construction (`SRE`, `RP`, `GLNN`, `RLNN` in [0,1]; `LRE` ≥ 1), so a misconfigured run is caught
before anything is pinned.

Per-feature, hardcoded golden against the fresh run, and against what Nyxus computes at the recipe
(`nyxus rel` is measured on the stored `*_AVE` slot; the per-angle assertion goes through `calc_ave`
and lands on the same number):

| feature | hardcoded | PyRadiomics 3.0.1 fresh | golden rel | nyxus rel | verdict |
|---|---|---|---|---|---|
| `3GLRLM_GLN` | 406.68709120394277 | 406.68709120394277 | 0 | 1.4e-16 | vetted |
| `3GLRLM_GLNN` | 0.09722976558135092 | 0.09722976558135092 | 0 | 0 | vetted |
| `3GLRLM_GLV` | 9.100102904831404 | 9.100102904831404 | 0 | 0 | vetted |
| `3GLRLM_HGLRE` | 130.25347348795043 | 130.25347348795043 | 0 | 0 | vetted |
| `3GLRLM_LGLRE` | 0.012578735424633676 | 0.012578735424633676 | 0 | 0 | vetted |
| `3GLRLM_LRE` | 1.5538285862328314 | 1.5538285862328314 | 0 | 2.9e-16 | vetted |
| `3GLRLM_LRHGLE` | 200.98033929654184 | 200.98033929654184 | 0 | 1.4e-16 | vetted |
| `3GLRLM_LRLGLE` | 0.01863138831176311 | 0.01863138831176311 | 0 | 0 | vetted |
| `3GLRLM_RE` | 4.228290966541947 | 4.228290966541947 | 0 | **3.9e-4** | vetted at `rel=5e-3` |
| `3GLRLM_RLN` | 3309.7814564084974 | 3309.7814564084974 | 0 | 1.4e-16 | vetted |
| `3GLRLM_RLNN` | 0.7807974007564221 | 0.7807974007564221 | 0 | 0 | vetted |
| `3GLRLM_RP` | 0.8714583333333334 | 0.8714583333333334 | 0 | 2.5e-16 | vetted |
| `3GLRLM_RV` | 0.19950155996777244 | 0.19950155996777244 | 0 | 1.4e-16 | vetted |
| `3GLRLM_SRE` | 0.9003824440228139 | 0.9003824440228139 | 0 | 2.5e-16 | vetted |
| `3GLRLM_SRHGLE` | 117.56903884692184 | 117.56903884692184 | 0 | 0 | vetted |
| `3GLRLM_SRLGLE` | 0.011465297979291003 | 0.011465297979291003 | 0 | 0 | vetted |

Each golden is the reference for both the base feature and its `_AVE` twin, so the sixteen rows above
carry all 32 registry rows.

## The gap was two rows, and it was an attribution problem

`test_3d_glrlm_ave_pyradiomics()` already asserted **14 of the 16** `_AVE` features against these
goldens and passed. The registry never recorded it: all sixteen `_AVE` rows read
`current_test = test_3d_glrlm_coverage.h;test_nyxus.py`, so the report counted the whole family as
vetted-without-an-oracle-test.

Two were genuinely unasserted in C++ — `3GLRLM_RP_AVE` and `3GLRLM_RE_AVE`, omitted from that
function's table while a comment claimed they were "already vetted elsewhere". They are now in it.
Both also read `oracle=mirp` while their base rows read `oracle=pyradiomics` for the same measured
quantity, with no notes on either — the identical incoherence found in 3D GLCM. No mirp run has ever
existed for 3D GLRLM in this tree: there is no `gen_glrlm3d_mirp.py`, no mirp goldens, and no
`test_3d_glrlm_mirp.h`. Both now carry `oracle=pyradiomics`.

To stop the list drifting again, `test_3d_glrlm_ave_pyradiomics()` now fails if any `3GLRLM_*`
feature the build exposes is missing from the golden table or from its own `_AVE` list. A feature
added to the family later cannot be silently unasserted while the test still passes over the entries
it has — which is exactly how these two were lost.

## The 10% band was eight orders of magnitude too loose

Measured, once the `_AVE` slots were being read:

| | Nyxus vs PyRadiomics |
|---|---|
| 15 of 16 features | ≤ 2.9e-16 (double precision; 6 of them exactly 0) |
| `3GLRLM_RE` | 3.9e-4 |

Every assertion in the file was banded at 10%. That is not a convention gap being absorbed — unlike
3D GLCM, whose asymmetric offset-1 cooc matrix genuinely differs from PyRadiomics' symmetric default,
GLRLM has no such divergence and lands on the tool exactly. A 10% band on an exact match would pass a
value wrong by a factor of 1.1.

Both the per-angle and the `_AVE` assertions are now held at `rel=1e-9`, with `3GLRLM_RE` at
`rel=5e-3`. Run entropy is the family's only sum over logarithms, evaluated through `fast_log10` with
an `EPSILON` guard; the same feature in the 2D family carries the same exception for the same reason
(measured 1.1e-3 there, 3.9e-4 here).

The pytest assertions carried a second, worse form of the same problem:
`np.isclose(..., rtol=1e-1, atol=1e-2)` on values as small as 0.0115 makes the effective band ±96%.
Those are tightened to match.

## 16 assertions that existed but never ran

`test_3d_glrlm_regression.h` was not `#include`d by `test_all.cc` and had no `TEST()`
registrations, for the same mechanical reason as its GLCM sibling: it carried its own **definition**
of `get_3d_segmented_phantom()`, which redefines the one in `test_3d_glcm_pyradiomics.h` inside the
single `test_all.cc` translation unit. It now forward-declares it, and its 16 tests are registered.

Wired in, 11 of the 16 failed. Two separate defects were underneath:

**The settings were incomplete.** The file set the generic `GREYDEPTH` only, but `D3_GLRLM_feature`
bins on `GLRLM_GREYDEPTH` (`3d_glrlm.cpp`). Left unset that defaults to 0, i.e. no binning at all, and
the features ran on raw intensities — `HGLRE` came out at 4.3e6 against the ~4e3 ceiling 64 grey
levels imply.

**The pins were unrecoverable.** No `GLRLM_GREYDEPTH` reproduces them. The best case is 5 of 16
within 10%, with residuals over 240%; the file's own comment says "Calculated at 100 grey levels" and
even that does not match. Against the current implementation at the recipe's binning:

| feature | old pin | regenerated | rel |
|---|---|---|---|
| `3GLRLM_SRE` | 0.84 | 0.84064583359383949 | 0.08% |
| `3GLRLM_RLNN` | 0.68 | 0.67562186885577269 | 0.6% |
| `3GLRLM_RLN` | 154513.0 | 132077.24363689235 | 14.5% |
| `3GLRLM_RP` | 0.83 | 0.70306894015499444 | 15.3% |
| `3GLRLM_RE` | 6.4 | 5.2580435451573697 | 17.8% |
| `3GLRLM_LGLRE` | 0.072 | 0.045266965539734721 | 37.1% |
| `3GLRLM_GLN` | 5811.0 | 9934.4008683562715 | 71.0% |
| `3GLRLM_GLV` | 254.9 | 30.204626150216143 | 88.2% |
| `3GLRLM_LRE` | 40.8 | 4.7269894613234555 | 88.4% |
| `3GLRLM_LRHGLE` | 5678.4 | 478.13693595593821 | 91.6% |
| `3GLRLM_HGLRE` | 1922.0 | 156.55736675409963 | 91.9% |
| `3GLRLM_SRHGLE` | 1771.8 | 141.01413107252367 | 92.0% |
| `3GLRLM_RV` | 34.9 | 2.5998881222894781 | 92.5% |
| `3GLRLM_LRLGLE` | 37.4 | 1.0699954232575346 | 97.1% |
| `3GLRLM_GLNN` | 0.026 | 0.051578758146456932 | 98.4% |
| `3GLRLM_SRLGLE` | 0.007 | 0.024058168696417533 | 243.7% |

They are regenerated from the current implementation — which the sibling file that *does* run vets
against PyRadiomics independently — at `GLRLM_GREYDEPTH=-20`, and the tolerance tightened from 10% to
`rel=1e-9`. `test_3d_glrlm_dump_regression()` regenerates them, so the next refresh is a filter away
rather than a hand transcription.

The helper also aggregated wrongly. It read `r.fvals[fcode][0]` under the comment *"we have just 1
value, no need to aggregate subfeatures"*, but `save_value` writes the full per-angle vector to that
slot — so the guard pinned one direction and let the other twelve drift unwatched. It now uses
`calc_ave` over the 13 angles, matching the GLCM regression file and the quantity the pins' names
imply.

## `3GLRLM_RP` leaves its mathematical bound — filed, not fixed here

Run Percentage is runs / voxels and cannot exceed 1. On the `ut_` segmented phantom, averaged over
the 13 angles, it does:

| `GLRLM_GREYDEPTH` | `RP` | `SRE` | `RLNN` | `GLNN` | `LRE` |
|---|---|---|---|---|---|
| −20 (binCount 20) | 0.703 | 0.841 | 0.676 | 0.052 | 4.73 |
| −100 (binCount 100) | 0.917 | 0.961 | 0.907 | 0.010 | 1.59 |
| 0 (unbinned) | 0.990 | 0.994 | 0.986 | 0.0006 | 1.04 |
| 32 | 0.819 | 0.766 | 0.556 | 0.052 | 33.5 |
| 64 | 0.940 | 0.846 | 0.688 | 0.030 | 27.0 |
| 100 | 0.987 | 0.872 | 0.736 | 0.023 | 25.0 |
| 128 | **1.007** | 0.882 | 0.757 | 0.020 | 24.3 |
| 256 | **1.045** | 0.901 | 0.794 | 0.014 | 23.0 |

It grows monotonically with the bin count and crosses 1 at positive grey depths ≥ 128 for the
13-angle average (≥ 64 for a single angle). Every other bounded feature in the family stays in range
across the whole sweep, so this is specific to `RP`. More runs than voxels is impossible, and the
monotone growth points at a denominator that does not track the binning.

**Not fixed here** — it is a behaviour change in a public feature and belongs on its own branch,
alongside a cheap invariant test (`RP ≤ 1`, `SRE ≤ 1`, `LRE ≥ 1`) so the class of defect cannot return
silently. That check is what found this one.

Both this branch's benchmarks use binCount binning, where `RP` is in range, and the `3GLRLM_RP_AVE`
registry row says so explicitly rather than claiming the feature holds at every config.

## Include hygiene and file-level observations

The family is three headers — `test_3d_glrlm_pyradiomics.h`, `test_3d_glrlm_regression.h`,
`test_3d_glrlm_coverage.h` — plus the `test_3d_glrlm_compatibility` case in
`tests/python/test_nyxus.py`. There is no `test_3d_glrlm_common.h`.

- **`test_3d_glrlm_pyradiomics.h` included `../src/nyx/raw_nifti.h` and referenced no symbol from
  it.** Removed; the build is unaffected. It also relied on `<string>`, `<unordered_set>` and
  `<vector>` transitively, and on `<tuple>` for the structured-binding decomposition of
  `get_3d_compat_phantom()`'s return; all four are now included directly.
- **`test_3d_glrlm_regression.h`** relied on `<iomanip>`, `<iostream>`, `<string>`, `<tuple>`,
  `<vector>` and `helpers/fsystem.h` (for `fs::exists`) transitively. All added.
- **`test_3d_glrlm_coverage.h` includes only `test_3d_coverage_common.h`**, from which it takes
  `ref_vals_map` and `std::vector` without naming either header. Left as-is deliberately: all eight
  3D `_coverage.h` files use the identical single-include form, `test_3d_coverage_common.h` is the
  shared fixture header SPEC §6.3.1 sanctions for exactly this, and changing one family's copy would
  break the uniformity for no functional gain.
- **Redundant duplicate assertions, both intentional and both now visible in the coverage CSV.** Each
  base feature is covered twice within `test_3d_glrlm_pyradiomics.h` (its own per-angle test, and the
  `_AVE` test's mapping table), and each `_AVE` feature is asserted a second time through the Python
  API by `test_3d_glrlm_compatibility` against the same goldens. The second pair is the one that
  matters operationally: the C++ and pytest tolerances must be re-tightened together, which is why
  the scanner now names non-oracle-suffixed coverage in the `Notes` column instead of dropping it.
- **`test_3d_glrlm_compatibility` does not conform to SPEC §6.2** (the suffix is neither an oracle
  token nor a kind). It is covered by `check_test_names.py`'s existing file-level exception for
  `test_nyxus.py` — *"88 API assertions across families; needs a by-family split"* — so it is tracked
  work, not a new violation. Splitting that file is out of scope for one family.

Two framework artifacts SPEC asks for do not exist for this family, and do not exist for any family
but GLCM: `tests/vetting/matrix/glrlm.md` (SPEC §5.1, the config-point matrix — only `matrix/glcm.md`
is present) and `tests/vetting/benchmarks.md` (SPEC §6.3, the benchmark registry — absent entirely).
Neither is created here: producing one for 3D GLRLM alone would leave the series inconsistent across
its seven other families. They are a repo-wide gap worth a PR of their own.

## Reproduction

```
# oracle goldens (conda env with pyradiomics 3.0.1, needs Python <= 3.9)
python tests/vetting/oracles/gen_glrlm3d_pyradiomics.py

# regression pins
runAllTests --gtest_filter=*3D_GLRLM_DUMP_REGRESSION*

# coverage artifact
python tests/vetting/audit/scan_glrlm3d_coverage.py [--check]
```

## Note for the remaining 3D families

Four sibling files still never execute — `test_3d_{gldm,glszm,ngtdm}_regression.h` (14, 16 and 5 test
functions) and `test_3d_firstorder_matlab.h` (35). Measured on this commit rather than assumed:

| file | `#include`d | `TEST()`s registered | assert body | phantom accessor |
|---|---|---|---|---|
| `test_3d_gldm_regression.h` | no | 0 of 14 | **live** | forward-declared |
| `test_3d_glszm_regression.h` | no | 0 of 16 | **live** | forward-declared |
| `test_3d_ngtdm_regression.h` | no | 0 of 5 | **live** | forward-declared |
| `test_3d_firstorder_matlab.h` | no | 0 of 35 | live (no `#if 0`) | forward-declared |
| `test_3d_ngldm_regression.h` | **yes** | **19 of 19** | **live** | forward-declared |

Two corrections to what the GLCM report and this series' notes previously said about these files, both
of which make the remaining work *smaller* than advertised:

- **They do not carry their own definition of `get_3d_segmented_phantom()`.** All five forward-declare
  it. The redefinition problem was specific to `test_3d_glcm_regression.h`; it was never the general
  cause. Wiring the rest in is an `#include` plus `TEST()` registrations, then fixing whatever the
  first run reports.
- **Their assertion bodies are not dead.** Each contains a disabled *old* body in `#if 0` followed by
  a live one — the same shape this file had — so the assertions are real once the file is reached.

`test_3d_ngldm_regression.h` in particular is **not** the "19 tests that assert nothing" case
described earlier: it is included, all 19 are registered, and the live `agrees_gt` at the end of its
helper does fire. Verified by negative control — perturbing one golden (`3NGLDM_DCENE` 0.14 → 0.99)
makes `TEST_3D_NGLDM_DCENE_REGRESSION` fail at that line. Its real open item is different and is
recorded in its own family's audit: the pinned values disagree with MIRP by up to an order of
magnitude, and 18 of its 19 rows are `status=regression` for that reason.
