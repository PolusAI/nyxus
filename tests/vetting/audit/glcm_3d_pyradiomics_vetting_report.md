# 3D GLCM vs PyRadiomics — vetting report

Closes the 29 `3GLCM_*_AVE` rows that read `status=vetted` with no in-tree oracle assertion, and
restores 26 assertions that existed in the tree but never executed.

## Tool and configuration

| | |
|---|---|
| Tool | PyRadiomics (`binCount: 20`, `interpolator: sitkBSpline`, `weightingNorm:` empty, `imageType: Original`) |
| Recipe | `glcm3d.pyradiomics_bincount20` |
| Fixture | the compat phantom, `compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii` |
| Nyxus config | `GREYDEPTH=100`, `IBSI=false`, `GLCM_GREYDEPTH=-20` (binCount binning), `GLCM_OFFSET=1`, `GLCM_SPARSEINTENS=true` |
| Test | `test_3d_glcm_pyradiomics.h` |
| Tolerance | `rel=1e-1`, matching the existing per-angle assertions on the same goldens |

## The gap was an attribution problem, not a missing measurement

`assert_3d_glcm_feature_pyradiomics()` ends in

```cpp
double atot = f.calc_ave (r.fvals[fcode]);          // average over the 13 3D angles
ASSERT_TRUE(agrees_gt(atot, glcm_3d_pyradiomics_ref_vals[fname], 10.));
```

PyRadiomics reports **one value per feature over its whole direction set**, which is exactly this
aggregation — so the right quantity was already being compared. But the assertion was booked against
the per-angle **base** feature (`3GLCM_ACOR`), while `save_value` separately writes the same number
to the feature the registry's gap rows actually name:

```cpp
copyfvals(fvals[(int)Feature3D::GLCM_ACOR], fvals_acor);      // per-angle vector
fvals[(int)Feature3D::GLCM_ACOR_AVE][0] = calc_ave(fvals_acor); // 3d_glcm.cpp:148
```

Nothing asserted the stored `*_AVE` features. The new `assert_3d_glcm_ave_feature_pyradiomics()`
reads `fvals[..._AVE][0]` rather than recomputing `calc_ave`, so a defect in how `save_value`
populates `*_AVE` now fails there and nowhere else. All 23 pass.

## The goldens had never been reproduced — and one was wrong

The family had **no generator**: `gen_glcm_{mirp,pyradiomics}.py` are 2D-only, so SPEC §6.4's
"generator script path" was unmet and the numbers rested on a CLI recipe written into a comment. That
comment also contradicted itself, naming the `ut_` phantom while every assertion uses the compat one.

`gen_glcm3d_pyradiomics.py` now runs PyRadiomics 3.0.1 against the compat phantom and re-verifies
every pinned golden. **19 of 23 reproduce bit-for-bit**, which confirms the table's origin. Five did
not:

| feature | pinned | PyRadiomics 3.0.1 | rel |
|---|---|---|---|
| `3GLCM_IDN` | 0.9822362042997563 | 0.9067759330416398 | 7.7% |
| `3GLCM_IDM` | 0.4040020605537021 | 0.3726945904589868 | 7.8% |
| `3GLCM_ID` | 0.47211428859469606 | 0.4459415317170447 | 5.5% |
| `3GLCM_IDMN` | 0.9822362042997563 | 0.9797065356412845 | 0.26% |
| `3GLCM_CORRELATION` | 0.4305477709920443 | 0.43309121847659515 | 0.59% |

`3GLCM_IDN` and `3GLCM_IDMN` were pinned to **byte-identical values** — the IDMN number pasted into
the IDN slot. PyRadiomics reports different numbers for the two. The error survived because the 10%
band absorbed 7.7% of it. The other four sit in the inverse-difference family, whose normalisation is
version-sensitive, so they are consistent with the table having been pinned by an earlier
PyRadiomics.

All five are re-pinned to the fresh run, and Nyxus still agrees with the corrected values inside the
same band. That band is not slack — it covers Nyxus' asymmetric offset-1 cooc matrix against
PyRadiomics' symmetric default — but this is a concrete demonstration that a 10% band cannot
distinguish a convention gap from a bad golden. Worth measuring the true per-feature residuals and
tightening in a follow-up.

## Six features vetted through an identity

PyRadiomics does not report `DIS`, `ENERGY`, `ENTROPY`, `HOM1`, `SUMVARIANCE` or `VARIANCE` under
their own names — `DIS` it deprecates outright as equivalent to `DifferenceAverage`. Each is
numerically identical to a twin that PyRadiomics does report:

| feature | twin |
|---|---|
| `3GLCM_DIS_AVE` | `3GLCM_DIFAVE_AVE` |
| `3GLCM_ENERGY_AVE` | `3GLCM_ASM_AVE` |
| `3GLCM_ENTROPY_AVE` | `3GLCM_JE_AVE` |
| `3GLCM_HOM1_AVE` | `3GLCM_ID_AVE` |
| `3GLCM_SUMVARIANCE_AVE` | `3GLCM_CLUTEND_AVE` |
| `3GLCM_VARIANCE_AVE` | `3GLCM_JVAR_AVE` |

The identity is not an assumption: `test_3d_glcm_equivalence_dump_pyradiomics()` already asserted it
at `1e-6` on the per-angle values, and `test_3d_glcm_ave_equivalence_pyradiomics()` now asserts it on
the stored `*_AVE` features **and** re-checks that the twin still matches its PyRadiomics golden, so
the chain cannot rot silently at either end.

## The `oracle=mirp` label on five rows was incoherent

`3GLCM_{DIFAVE,DIS,IDN,IDMN,INFOMEAS2}_AVE` read `oracle=mirp` while their base rows read
`oracle=pyradiomics` — the same measured quantity attributed to two different tools, with no notes on
either. No mirp run has ever existed for 3D GLCM in this tree: there is no `gen_glcm3d_mirp.py`, no
mirp goldens, and no `test_3d_glcm_mirp.h`. The five now carry `oracle=pyradiomics`, which is the
tool whose golden the assertion actually compares against.

## 26 assertions that existed but never ran

**`test_3d_glcm_regression.h` (25 tests).** Not `#include`d by `test_all.cc`, with zero `TEST()`
registrations. The cause is mechanical, not an oversight: the file carried its own **definition** of
`get_3d_segmented_phantom()`, which redefines the one in `test_3d_glcm_pyradiomics.h` inside the
single `test_all.cc` translation unit. The live 3D headers forward-declare it instead; this file now
does too, and its 25 tests are registered.

Wired in, all 25 failed. The pins were not merely stale — several were **impossible**:

| feature | pinned | computed | bound |
|---|---|---|---|
| `3GLCM_ID` | 2.5 | 0.277 | ≤ 1 |
| `3GLCM_IDM` | 2.4 | 0.193 | ≤ 1 |
| `3GLCM_IDN` | 3.8 | 0.940 | ≤ 1 |
| `3GLCM_IDMN` | 3.9 | 0.990 | ≤ 1 |
| `3GLCM_JMAX` | 1.86 | 0.0083 | ≤ 1 |

`ID`, `IDM`, `IDN`, `IDMN` and `JMAX` are bounded in [0,1] by construction; a probability cannot be
1.86. The old pins also broke the `SUMVARIANCE == CLUTEND` identity the family asserts elsewhere
(18057.4 against 18057.0), while the current values hold it exactly and give `DIS == DIFAVE` to
3.6e-15. Every current value is in range.

They are therefore regenerated from the current implementation — which is independently vetted
against PyRadiomics on the compat phantom by the sibling file that does run — and the tolerance
tightened from 10% to `rel=1e-9`, since these are Nyxus' own values pinned to full precision.
`test_3d_glcm_dump_regression()` regenerates them, so the next refresh is a filter away rather than a
hand transcription.

**`test_3d_glcm_jvar_pyradiomics` (1 test).** A complete assertion with no `TEST()` calling it
(`not_covered.md` §B.2). Registered; it passes.

## Note for the remaining 3D families

Five sibling files are in the same unreachable state — `test_3d_{gldm,glrlm,glszm,ngtdm}_regression.h`
and `test_3d_firstorder_matlab.h` — each carrying its own definition of `get_3d_segmented_phantom()`.
Unlike the GLCM one, the four `_regression` files have their assertion bodies wrapped in `#if 0`, so
wiring them in yields tests that assert nothing; the body has to be restored first. Worth knowing
before PRs 8-10 repeat this exercise.

`test_3d_ngldm_regression.h` is a third variant: it *is* included and its 19 tests *do* run, but its
assert body is `#if 0` too — so those 19 have been passing without checking anything.
