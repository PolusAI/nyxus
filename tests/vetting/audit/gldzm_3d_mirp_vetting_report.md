# 3D GLDZM vs the MIRP oracle — vetting report

All 18 rows of this family read `status=regression` with no oracle named, and this pass does not
promote any of them. What it establishes is *why* they cannot be promoted: Nyxus does not compute
the IBSI GLDZM definition, and the gap is a defect rather than a tolerance.

## Reproduction

| | |
|---|---|
| generator | `tests/vetting/oracles/gen_gldzm3d_mirp.py` |
| oracle | MIRP 2.6.0 (numpy 2.4.6, scipy 1.17.1, Python 3.11), env `nyxus_mirp` |
| fixture | `tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57, 274432 voxels |
| recipe | `gldzm3d.mirp_fbn64` — `by_slice=False`, `fixed_bin_number` n=64; Nyxus side `GREYDEPTH=64`, `IBSI=false` |

```
python tests/vetting/oracles/gen_gldzm3d_mirp.py
```

MIRP implements IBSI GLDZM. PyRadiomics has no GLDZM at all, so MIRP is the only mainstream oracle
available for this family.

## Nyxus disagrees with MIRP on every feature MIRP computes

Nyxus values are the full-precision pins now in `test_3d_gldzm_regression.h`.

| feature | Nyxus | MIRP | Nyxus / MIRP |
|---|---:|---:|---:|
| `LDHGLE` | 734618.35720259824 | 10881.7095947 | **67.5×** |
| `LDE` | 314.01248309662088 | 11.2315087359 | **28.0×** |
| `ZDV` | 79.723412707174901 | 3.24563048183 | **24.6×** |
| `SDLGLE` | 1.8362515436029654e-05 | 0.00021623223212 | 0.085× |
| `SDE` | 0.022387420258025731 | 0.381988292648 | 0.059× |
| `SDHGLE` | 61.230746106573264 | 1098.11250725 | 0.056× |
| `ZDNU` | 4330.2817177741472 | 14488.2767411 | 0.299× |
| `ZDNUN` | 0.033848043255251946 | 0.195623622655 | 0.173× |
| `LDLGLE` | 0.16729167507144088 | 0.0347909820736 | 4.81× |
| `LGLZE` | 0.0005581993242951194 | 0.00149522435841 | 0.373× |
| `GLNU` | 3435.1800942680934 | 1433.44562664 | 2.40× |
| `GLV` | 111.77220626552925 | 227.689486006 | 0.491× |
| `ZP` | 0.46617376982276121 | 0.269873775653 | 1.73× |
| `GLNUN` | 0.026851399515903585 | 0.0193546707709 | 1.39× |
| `ZDE` | 10.230312642315166 | 7.50462066483 | 1.36× |
| `HGLZE` | 2342.4734665801629 | 1920.82497097 | 1.22× |

## The definition is reachable — so the gap is Nyxus'

A disagreement with one tool proves nothing on its own; the tool could be the odd one out. So the
generator recomputes all 16 features from scratch, independently of both implementations: zones are
the **26-connected components** of each discretised grey level, and the distance to the ROI border
is a **city-block distance transform**.

That reproduces MIRP to **rel 3.2e-16** — machine precision, on all 16. A different route to the
same mathematics agreeing to the last bit is what licenses the conclusion that the IBSI definition
is well defined on this fixture and that Nyxus is not computing it.

## Which part of Nyxus is wrong: the zone map, not the distance

The generator answers this too, by recomputing the same features a third time with **Nyxus'
straight-ray distance** substituted for the distance transform, holding the zone map fixed.

Worst change: **1.5e-2** (`dzm_zd_var`); most features move under 1e-3. Over the ROI the two metrics
give mean distances of 5.5198 and 5.5418.

So the distance metric accounts for about one percent. It cannot explain a 67× disagreement. The
zone map can, and the zone counts show it directly:

| zone decomposition | zones | `ZP` |
|---|---:|---:|
| 6-connected | 163,888 | 0.597 |
| 18-connected | 95,148 | 0.347 |
| **26-connected (= MIRP)** | **74,062** | **0.2699** |
| **Nyxus** | **127,933** | **0.46617376982276121** |

Nyxus' count matches **no** connectivity. It sits between the 6- and 18-connected values, so the
zone-growing search in `prepare_GLDZM_matrix_kit` is not producing same-level connected components
under any neighbourhood definition — which is a stronger statement than "it uses the wrong
connectivity", and it is why correcting the neighbour list alone does not fix the family.

## Three defects, none fixed here

1. **The zone search does not produce connected components.** Above. `src/nyx/features/3d_gldzm.cpp`
   walks six axis neighbours with a parent stack; the resulting decomposition matches neither
   6- nor 18- nor 26-connectivity. Fixing it is an implementation job, not a vetting one, and it
   changes every value in this family.

2. **`dist2border` never scans the z axis.** `3d_gldzm.cpp` scans left, right, up and down — all
   within one z-slice — and takes the minimum of four. The 2D implementation in `gldzm.cpp` scans
   the same four, which for 2D is both axes and therefore complete; the 3D version is that function
   with a `z` parameter threaded through and never scanned, so a voxel one slice from the ROI
   surface is scored by its in-plane rays alone. Real, and independent of defect 1, but worth about
   one percent on this fixture rather than an order of magnitude.

3. **`3GLDZM_ZDM` was pinned at 222 and never asserted.** The golden sat in the reference table with
   no test function and no `TEST()` registration, so it could not fail. Nyxus computes
   **15.306504185784746**. The header's own geometric argument was already sufficient to reject 222:
   the same distribution's variance is 79.72, so its standard deviation is about 8.9, and a mean of
   222 is impossible. Now registered, and pinned to the value Nyxus actually produces.

`3GLDZM_GLM` and `3GLDZM_ZDM` have no MIRP or IBSI counterpart at all — MIRP emits no `dzm_gl_mean`
or `dzm_zd_mean` — so they stay regression-only however defects 1 and 2 are resolved.

## What this pass changed

Only what is independent of the defects:

- `3GLDZM_ZDM` gained the missing test function and registration, taking the family from 17 executed
  assertions to 18.
- All 18 goldens re-pinned at full `%.17g` precision. Seventeen of them were the same values rounded
  to between two and five significant figures.
- The band moved from `agrees_gt(..., 10.)` — a **±10%** band, since the third argument is a divisor
  — to `rel=1e-9`. A drift guard compares the program against its own output, so movement is the
  only thing it can detect, and at ±10% with two-significant-figure pins it could not have detected
  a value drifting by a third.

The values remain wrong with respect to IBSI. They are pinned as a change detector for the eventual
fix, and the registry says so; nothing here promotes a row.

## The retired coverage sweep, and include hygiene

`test_3d_gldzm_coverage.h` instantiated the two generic `TEST_P` suites of
`test_3d_coverage_common.h` over the family's 18 features and published a second pin table,
`gldzm_3d_regression_coverage_ref_vals`. It is gone. What was measured before deleting it:

- **The two tables now hold the same 18 keys, 16 of them bit-identical.** The two that differ,
  `3GLDZM_GLV` and `3GLDZM_ZDE`, differ in the last decimal digit — 1.8e-16 and 2.0e-16 relative,
  one ULP, and seven orders of magnitude inside the `rel=1e-9` band both copies were asserted at. No
  assertion in the tree can distinguish them.
- **They were not duplicates before this pass, and that is why the retirement could not be cut from
  `main`.** The regression copy carried the same quantities rounded to between two and five
  significant figures, and `3GLDZM_ZDM` read `222` against the sweep's `15.306504185784746` — a
  factor of 14.5. Nothing caught it because `3GLDZM_ZDM` had no test function and no registration:
  the pin existed and was read by nothing. Re-pinning at full precision and registering that
  assertion are what made the two tables comparable in the first place.
- Both are taken at the same recipe. The sweep runs `GREYDEPTH=64`, `IBSI=false`; so does this
  file's helper. `make_3d_coverage_settings()` additionally sets the GLCM/GLDM/GLRLM/GLSZM/NGTDM
  grey depths and the NGTDM radius, none of which GLDZM reads, so the two configurations are
  equivalent for this family.
- The `GLDZM_WITH_3P_EMBEDDED_GT` half instantiated **zero** cases: no GLDZM table appears in
  `externally_vetted_3d_feature_names()`, because the family has no oracle-backed feature at all —
  which is the subject of the rest of this report.
- All 18 features have an individually named `*_regression` test and a `TEST()` registration, so
  nothing had to be ported.

**What the sweep additionally checked, and where it lives now.** The name-resolves-and-code-matches
step is done by every named test (`find_3D_FeatureByString` plus an assert on the returned code).
The one-provider-per-`Feature3D`-code step is done by `FeatureManager::check_11_correspondence()`,
which `test_feature_manager_mechanics.h` exercises through `fm.compile()`.

**One step has no counterpart and is genuinely dropped**, here as in the three families retired
before this one: the sweep also ran all nine 3D calculators once over the phantom and asserted that
each feature's slot in the global value table came back non-empty with at least one finite value —
i.e. that `save_value()` writes it. The named tests build their own ROI and read `r.fvals[fcode][0]`
directly, so none of them exercise that path. Worth stating plainly rather than implying the port is
lossless. Its practical reach was narrower than it sounds: `make_3d_coverage_settings()` hand-sets
every grey depth and the NGTDM radius, so the check only ever covered the calculators under a
correctly configured `Fsettings`, never the defaults that `compile_feature_settings()` produces.

The completeness guard in `test_3d_coverage_common.h` reads this family's pins straight off
`gldzm_3d_regression_ref_vals`, so the migration cost exactly one `add_keys()` line. Deleting a pin
from the table now fails `TEST_3D_FEATURE_COVERAGE_COUNTS` by feature name. Note what that guard
does **not** do: `feature_3d_cases()` classifies all 213 public features whether or not any family
instantiates the suites, so the `94`/`119` counts are unmoved by a retirement and cannot detect one.

**Include hygiene.** Being the family's fixture as well as its table, `test_3d_gldzm_regression.h`
used `Fsettings`/`NyxSetting`, `SlideProps`/`scan_slide_props`, the four `globals.h` ROI-gathering
functions, `fs::exists` and `agrees_gt` without including `feature_settings.h`, `slideprops.h`,
`globals.h`, `helpers/fsystem.h` or `test_main_nyxus.h` — every one reached it transitively. All are
now direct. The pin table is also `const` and read through `.at()`: with the coverage copy gone this
file is the family's only pin table and `test_3d_coverage_common.h` reads its keys, so an
`operator[]` default-insert would have both passed a bogus assertion against a 0 golden and added a
phantom feature name to the individually-pinned set.
